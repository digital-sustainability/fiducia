from pydantic import BaseModel

from typing import List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field


class DateGranularity(str, Enum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    UNKNOWN = "unknown"


class DateMention(BaseModel):
    """Represents a date mentioned in the text."""

    id: str = Field(..., description="A unique identifier for this date mention.")
    text: str = Field(..., description="The original text of the date mention.")
    iso_start: str = Field(..., description="The normalized date in YYYY, YYYY-MM, or YYYY-MM-DD format.")
    iso_end: Optional[str] = Field(
        default=None, description="The normalized end date if the mention represents a range."
    )
    granularity: DateGranularity = DateGranularity.UNKNOWN
    confidence: float = Field(..., ge=0.0, le=1.0, description="The model's confidence in the extraction.")


class EventType(str, Enum):
    FILING = "filing"  # Klageeinreichung / Gesuchseinreichung
    HEARING = "hearing"  # Verhandlung
    JUDGMENT = "judgment"  # Urteil
    ORDER = "order"  # Verfügung / Beschluss
    APPEAL = "appeal"  # Berufung / Beschwerde
    MOTION = "motion"  # Antrag
    DEADLINE = "deadline"  # Frist
    SETTLEMENT = "settlement"  # Vergleich
    CONCILIATION_REQUEST = "conciliation_request"  # Schlichtungsgesuch
    AUTHORIZATION_TO_PROCEED = "authorization_to_proceed"  # Klagebewilligung
    OBJECTION = "objection"  # Einsprache
    INDICTMENT = "indictment"  # Anklageschrift
    PENALTY_ORDER = "penalty_order"  # Strafbefehl
    OTHER = "other"


class ParticipantRole(str, Enum):
    PLAINTIFF = "plaintiff"  # Kläger
    DEFENDANT = "defendant"  # Beklagter
    JUDGE = "judge"  # Richter
    LAWYER = "lawyer"  # Anwalt
    WITNESS = "witness"  # Zeuge
    PROSECUTOR = "prosecutor"  # Staatsanwalt
    ACCUSED = "accused"  # Beschuldigte Person
    APPELLANT = "appellant"  # Beschwerdeführer
    RESPONDENT = "respondent"  # Beschwerdegegner
    EXPERT = "expert"  # Sachverständiger
    OTHER = "other"


class ParticipantMention(BaseModel):
    """Represents a participant (person or organization) mentioned in the text."""

    id: str = Field(..., description="A unique identifier for this participant mention.")
    name: str = Field(..., description="The name of the participant.")
    role: ParticipantRole = Field(
        default=ParticipantRole.OTHER, description="The role of the participant in the case."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="The model's confidence in the extraction.")


class EventMention(BaseModel):
    """Represents a legal event mentioned in the text."""

    id: str = Field(..., description="A unique identifier for this event mention.")
    label: str = Field(..., description="A concise description of the event.")
    event_type: EventType = EventType.OTHER
    date_id: str = Field(..., description="The ID of the date mention associated with this event.")
    participant_ids: List[str] = Field(
        default_factory=list, description="A list of participant IDs involved in this event."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="The model's confidence in the extraction.")


class LegalChunkAnalysis(BaseModel):
    """A container for all entities extracted from a single chunk of a legal document."""

    dates: List[DateMention] = Field(default_factory=list)
    events: List[EventMention] = Field(default_factory=list)
    participants: List[ParticipantMention] = Field(default_factory=list)
