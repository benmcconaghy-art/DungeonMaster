"""Campaign endpoints (minimal Phase 2 surface).

Phase 2 only needs enough campaign API to bootstrap a single-player
session: create the campaign and add the creating user as the owner
member. List/show/invite/join from spec §11 land in Phase 4 with
multiplayer.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.db import models
from app.deps import CurrentUser, DbSession

router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])


class CreateCampaignRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class CampaignResponse(BaseModel):
    id: str
    name: str
    owner_id: str
    ruleset: str
    created_at: str


def _campaign_to_response(campaign: models.Campaign) -> CampaignResponse:
    return CampaignResponse(
        id=campaign.id,
        name=campaign.name,
        owner_id=campaign.owner_id,
        ruleset=campaign.ruleset,
        created_at=campaign.created_at,
    )


@router.post("", response_model=CampaignResponse, status_code=status.HTTP_201_CREATED)
async def create_campaign(
    payload: CreateCampaignRequest,
    user: CurrentUser,
    db: DbSession,
) -> CampaignResponse:
    """Create a campaign owned by the current user."""

    campaign = models.Campaign(name=payload.name, owner_id=user.id)
    db.add(campaign)
    await db.flush()
    # Owner row in campaign_members so role-checks work uniformly
    # (campaign_members is the authoritative membership table; the
    # owner_id column is just a quick pointer).
    db.add(
        models.CampaignMember(
            campaign_id=campaign.id,
            user_id=user.id,
            role="owner",
        )
    )
    await db.commit()
    await db.refresh(campaign)
    return _campaign_to_response(campaign)


__all__ = ["CampaignResponse", "CreateCampaignRequest", "router"]
