from dataclasses import dataclass, field


@dataclass(frozen=True)
class UserContext:
    user_id: str
    clearance: str  # "engineering" or "leadership"
    teams: frozenset[str] = field(default_factory=frozenset)


class RuleACL:
    def visible(self, doc: dict, user: UserContext) -> bool:
        # Engineering cannot see leadership-only documents.
        if user.clearance == "engineering" and doc.get("access_level") == "leadership":
            return False
        # Leadership sees everything accessible.
        if user.clearance == "leadership":
            return True
        # Engineering: DEFAULT-DENY on empty teams.
        if not user.teams:
            return False
        if not doc.get("team"):
            return False
        return doc["team"] in user.teams