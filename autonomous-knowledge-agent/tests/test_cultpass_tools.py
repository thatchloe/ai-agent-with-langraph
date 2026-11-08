"""Business rules of the CultPass operation tools (external DB)."""
from agentic.tools import cultpass_tools as ct


def test_customer_profile_includes_subscription_and_quota(dbs):
    profile = ct.get_customer_profile("f556c0")
    assert profile["ok"] and profile["full_name"] == "Bob Stone"
    assert profile["subscription"]["tier"] == "premium"
    assert profile["subscription"]["reservations_this_month"] == 3
    assert profile["subscription"]["remaining_quota"] == 5


def test_unknown_customer(dbs):
    assert ct.get_customer_profile("nope")["error"] == "user_not_found"


def test_list_reservations(dbs):
    titles = {r["title"] for r in ct.list_reservations("f556c0")["reservations"]}
    assert "Modern Art at MASP" in titles and "Jazz Brunch in Pinheiros" in titles


def test_search_experiences_by_keyword_and_location(dbs):
    assert ct.search_experiences("samba")["experiences"][0]["title"] == "Samba Night at Lapa"
    rio = ct.search_experiences(location="Rio")["experiences"]
    assert rio and all("Rio" in e["location"] for e in rio)
    assert all(e["slots_available"] > 0 for e in ct.search_experiences()["experiences"])


def test_reserve_success_decrements_slots_and_quota(dbs):
    before = ct.search_experiences("samba")["experiences"][0]["slots_available"]
    result = ct.reserve_experience("f556c0", "exp-3")
    assert result["ok"] and result["remaining_quota"] == 4
    assert ct.search_experiences("samba")["experiences"][0]["slots_available"] == before - 1
    assert ct.reserve_experience("f556c0", "exp-3")["error"] == "already_reserved"


def test_reserve_rules(dbs):
    assert ct.reserve_experience("a4ab87", "exp-3")["error"] == "account_blocked"
    assert ct.reserve_experience("f1f10d", "exp-3")["error"] == "subscription_inactive"
    assert ct.reserve_experience("88382b", "exp-0")["error"] == "premium_requires_upgrade"
    assert ct.reserve_experience("88382b", "exp-full")["error"] == "sold_out"
    assert ct.reserve_experience("88382b", "missing")["error"] == "experience_not_found"
    assert ct.reserve_experience("e6376d", "exp-1")["ok"]          # Frank: 2nd of 2
    assert ct.reserve_experience("e6376d", "exp-5")["error"] == "quota_exceeded"


def test_cancel_reservation_rules(dbs):
    assert ct.cancel_reservation("88382b", "r-bob-2")["error"] == "reservation_not_found"  # not hers
    assert ct.cancel_reservation("f556c0", "r-bob-late")["error"] == "late_cancellation"
    ok = ct.cancel_reservation("f556c0", "r-bob-2")
    assert ok["ok"] and ok["title"] == "Modern Art at MASP"
    assert ct.cancel_reservation("f556c0", "r-bob-2")["error"] == "not_active"
    assert ct.get_customer_profile("f556c0")["subscription"]["remaining_quota"] == 6


def test_search_with_neighbourhood_as_location_falls_back_to_keywords(dbs):
    """Regression (found with the live LLM): location='Lapa' is a neighbourhood, not a stored location."""
    found = ct.search_experiences("Samba Night", location="Lapa")["experiences"]
    assert found and found[0]["title"] == "Samba Night at Lapa"


def test_empty_search_explains_it_is_not_sold_out_and_suggests_catalogue(dbs):
    """Regression (found in a live run): 'music events' matched nothing although Samba Night exists."""
    result = ct.search_experiences("music")
    assert result["count"] == 0 and "does NOT mean" in result["message"]
    assert "Samba Night at Lapa" in [e["title"] for e in result["suggestions"]]
    assert all(e["slots_available"] > 0 for e in result["suggestions"])
