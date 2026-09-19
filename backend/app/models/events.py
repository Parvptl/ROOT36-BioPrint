"""The browser-to-server behavioural event contract.

This is the only behavioural data the client is permitted to send. It carries
*no* score, *no* feature vector and *no* decision: the browser is a sensor, the
server is the trust boundary. Everything downstream is recomputed here.

Privacy rule enforced by the schema itself
------------------------------------------
`code` (the physical key identity, e.g. "KeyT") may only be populated for
contexts whose text is public — the username field and the server-generated
challenge phrase. In the password context `code` MUST be null and the client
sends only a coarse `key_class`, in which every printable character collapses
to `char`. The server rejects a payload that violates this, so a modified or
malicious client cannot exfiltrate password characters through this endpoint
even if it wants to.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Where an event happened. Used for per-field features and for the privacy rule.
FieldContext = Literal["username", "password", "phrase", "page"]

# Coarse key categories. `char` deliberately covers every printable character
# so the password context leaks no character-class information (not even
# "this position was a digit").
KeyClass = Literal["char", "backspace", "shift_left", "shift_right", "nav", "enter", "other"]

# How focus arrived in a field. Keyboard-vs-pointer navigation habit is a
# stable personal trait and costs nothing to record.
FocusVia = Literal["tab", "pointer", "programmatic", "unknown"]

MAX_EVENTS = 20_000  # bounds memory and extraction cost for one attempt
MAX_CODE_LEN = 32


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields are a red flag, not a shrug

    # Milliseconds since the collector started, from performance.now(). Monotonic
    # and unaffected by wall-clock changes, which matters for timing features.
    t: float = Field(ge=0, le=3_600_000)
    trusted: bool  # event.isTrusted; one low-weight automation signal among many


class KeyEventIn(_Base):
    type: Literal["keydown", "keyup"]
    ctx: FieldContext
    key_class: KeyClass
    code: str | None = Field(default=None, max_length=MAX_CODE_LEN)
    repeat: bool = False

    @field_validator("code")
    @classmethod
    def _code_charset(cls, v: str | None) -> str | None:
        # KeyboardEvent.code values are ASCII alphanumerics only. Anything else
        # is either a broken client or an injection attempt.
        if v is not None and not v.isalnum():
            raise ValueError("code must be alphanumeric")
        return v

    @model_validator(mode="after")
    def _no_password_characters(self) -> "KeyEventIn":
        if self.ctx == "password" and self.code is not None:
            raise ValueError("code must be null for password-context key events")
        return self


class PointerMoveIn(_Base):
    type: Literal["pointermove"]
    x: float = Field(ge=-10_000, le=10_000)
    y: float = Field(ge=-10_000, le=10_000)


class PointerButtonIn(_Base):
    type: Literal["pointerdown", "pointerup"]
    ctx: FieldContext
    x: float = Field(ge=-10_000, le=10_000)
    y: float = Field(ge=-10_000, le=10_000)
    button: int = Field(ge=0, le=4)


class FocusEventIn(_Base):
    type: Literal["focus", "blur"]
    ctx: FieldContext
    via: FocusVia = "unknown"


class InputEventIn(_Base):
    """Fired by the browser whenever a field's value changes.

    Kept because an `input` with no preceding `keydown` is the signature of
    `element.value = "..."` — the cheapest and most common scripted fill.
    Only the resulting length is recorded, never the content.
    """

    type: Literal["input"]
    ctx: FieldContext
    length: int = Field(ge=0, le=10_000)


class SimpleEventIn(_Base):
    type: Literal["submit", "paste"]
    ctx: FieldContext


BehaviorEvent = Annotated[
    Union[
        KeyEventIn,
        PointerMoveIn,
        PointerButtonIn,
        FocusEventIn,
        InputEventIn,
        SimpleEventIn,
    ],
    Field(discriminator="type"),
]


class ClientMeta(BaseModel):
    """Environment facts the client volunteers.

    Treated as *hints only*. `webdriver` in particular is trivially spoofed and
    is deliberately given low weight by the automation detector; it is recorded
    for the audit trail, not used as a gate.
    """

    model_config = ConfigDict(extra="forbid")

    webdriver: bool = False
    pointer_type: Literal["mouse", "touch", "pen", "none"] = "mouse"
    screen_w: int = Field(default=0, ge=0, le=20_000)
    screen_h: int = Field(default=0, ge=0, le=20_000)
    hardware_concurrency: int = Field(default=0, ge=0, le=1024)


class BehaviorSessionIn(BaseModel):
    """One captured interaction, bound to one single-use challenge nonce."""

    model_config = ConfigDict(extra="forbid")

    nonce: str = Field(min_length=16, max_length=128)
    # What the user actually typed into the challenge field. The phrase is
    # server-generated and public, so echoing it back leaks nothing; the server
    # compares it against the issued phrase to defeat replay of an old capture.
    phrase_typed: str = Field(default="", max_length=500)
    started_at_ms: float = Field(ge=0)
    events: list[BehaviorEvent] = Field(min_length=1, max_length=MAX_EVENTS)
    meta: ClientMeta = Field(default_factory=ClientMeta)

    @field_validator("events")
    @classmethod
    def _timestamps_non_decreasing(cls, events: list) -> list:
        # The collector appends in dispatch order, so a decreasing timestamp
        # means the stream was reordered or hand-assembled. Caught here as a
        # malformed payload rather than silently producing garbage features.
        last = -1.0
        for ev in events:
            if ev.t < last:
                raise ValueError("event timestamps must be non-decreasing")
            last = ev.t
        return events
