"""LLM prompts for the pipeline's three litellm call sites.

Kept out of the stage modules so prompt edits don't require touching the
logic that calls them, and so all prompt text lives in one place to review.
"""

from __future__ import annotations

# stages/condense.py — rewrite_query_llm() (Stage 0, Phase 3 multi-turn rewrite)
QUERY_REWRITE_PROMPT = """Given a conversation, output the user's CURRENT need as a short standalone search query. Output the query and nothing else.

Rules:
- Current means the most recent turns. If an earlier need was resolved, abandoned, or replaced, ignore it.
- Fold in constraints from earlier turns that still apply to the current need: budget, location, size, brand preferences, exclusions ("not X").
- Resolve pronouns and references: "are they waterproof" after discussing trail runners -> "waterproof trail running shoes".
- Use the user's words where possible. Do not add constraints, qualifiers, or intent the user did not express.
- Keep it under ~10 words, like a search engine query.
- Output exactly NONE if the current turn has no concrete need a search could serve: greetings, thanks, emotional support, opinion chat, creative requests, coding help, or if the user just completed/decided and is wrapping up.

Examples:

User: I'm looking for running shoes, budget is $120 max
User: actually I mostly run on trails
-> trail running shoes under $120

User: best espresso machines?
Assistant: [recommendations]
User: thanks, ordered the breville!
-> NONE

User: how do I fix this pandas KeyError
-> NONE

User: we're planning a trip to Lisbon in March
User: what about somewhere to stay near the old town
-> hotels near Alfama old town Lisbon March

User: I need a gift for my dad, he's into fishing
User: something under $50
-> fishing gift for dad under $50

User: my washing machine is leaking
User: ugh, I think it's dead. what's a reliable replacement?
-> reliable washing machine
"""

# stages/intent.py — LLMIntentClassifier.classify() (Gate A)
INTENT_FEW_SHOT_PROMPT = """You grade user queries for commercial intent on a 0-3 scale.

The core question: how close is this user to a purchase decision an ad could influence?

0 = No commercial intent. Wants information, help, or content — not a product/service. Includes homework, definitions, code, emotional support, and troubleshooting/support for a product they already own.
1 = Commercial topic, no active need. Learning about a product or service category out of curiosity or education, with no sign of shopping.
2 = Active commercial research. Comparing or evaluating options: "best X", "X vs Y", attribute-constrained product queries, "is X worth it", cost/price research at the category level.
3 = Transactional or local intent. Ready to buy, book, or visit: "buy", "near me", "open now", "deals", "coupon", price check on a specific model, local service lookups.

Rules:
- Grade the query as written. Do not imagine a more commercial version of it.
- Purchase-stage signals outweigh nouns. A brand or product name alone does not make a query commercial.
- Troubleshooting an owned product is 0, even if a brand is named.
- If genuinely torn between two grades, pick the lower one.

Examples:
"explain what a derivative is" -> {"grade": 0, "reason": "educational question, no product"}
"my dyson v11 won't charge" -> {"grade": 0, "reason": "post-purchase troubleshooting, not shopping"}
"write a haiku about autumn" -> {"grade": 0, "reason": "content generation request"}
"how do index funds work" -> {"grade": 1, "reason": "learning about a financial category, no active need"}
"what is mirrorless vs dslr" -> {"grade": 1, "reason": "category education without shopping signals"}
"is a standing desk worth it" -> {"grade": 2, "reason": "pre-purchase evaluation of a product"}
"bathroom fan without light" -> {"grade": 2, "reason": "attribute-constrained product search implies shopping"}
"sony wh-1000xm5 vs bose qc ultra" -> {"grade": 2, "reason": "direct product comparison"}
"how much do solar panels cost" -> {"grade": 2, "reason": "cost research on a purchase category"}
"iphone 15 pro price" -> {"grade": 3, "reason": "price check on a specific model signals purchase readiness"}
"best pizza delivery open now" -> {"grade": 3, "reason": "immediate local transactional intent"}
"emergency plumber austin" -> {"grade": 3, "reason": "urgent local service need"}

Reply with ONLY the JSON object, no markdown fences: {"grade": <0-3>, "reason": "<one short sentence>"}"""

# stages/judge.py — LLMJudge.judge() (Stage C)
JUDGE_PROMPT = """You are the final quality gate for ad placements inside an AI assistant conversation. Given the conversation and a candidate ad, decide whether showing this ad to this user right now is appropriate.

Approve (show: true) only if ALL of these hold:
1. Relevance: the ad addresses the user's actual current need — not just a keyword overlap with something they said. The need must be live in the most recent turns, not something resolved or abandoned earlier in the conversation.
2. Category match: the ad's product/service category is what the user is looking for. An adjacent category is a reject (user wants running shoes -> insoles ad is a reject).
3. Constraint compatibility: the ad does not contradict explicit user constraints (budget, location, dietary needs, "not brand X", features they ruled out).
4. Context appropriateness: nothing in the conversation makes advertising inappropriate right now.

Always reject (show: false) when any of these apply:
- The user is in distress: health scare, grief, financial hardship, crisis, emotional support. This includes ads that "solve" the distress (debt-relief ad on a bankruptcy question is still a reject).
- The conversation involves a minor's needs in a sensitive area.
- The ad exploits a vulnerability the user disclosed (addiction, disordered eating, gambling, desperation).
- The user already made their decision or completed the purchase — the ad would arrive too late.
- The ad contradicts advice the assistant just gave (assistant said "you don't need a VPN for this" -> VPN ad is a reject).
- The user explicitly asked not to see ads or recommendations.

Judge only what is in the conversation. Do not invent user needs the ad might satisfy.
When uncertain, reject. A missed impression is cheap; a jarring or exploitative ad damages trust in the entire product.

Examples:
Conversation: user comparing hiking backpacks under $150. Ad: 55L hiking backpack, $129.
-> {"show": true, "reason": "matches the live need and stays within the stated budget"}

Conversation: user comparing hiking backpacks under $150. Ad: premium backpack, $310.
-> {"show": false, "reason": "more than double the user's explicit budget"}

Conversation: user asked how to talk to their kid about their divorce. Ad: family law firm.
-> {"show": false, "reason": "emotionally sensitive parenting moment; ad exploits the situation"}

Conversation: user debugging Python, mentioned they're doing this for their e-commerce site. Ad: e-commerce platform.
-> {"show": false, "reason": "user's live need is debugging help; e-commerce was incidental context"}

Conversation: user asked for pizza near them, assistant listed options, user said "ordered from the first one, thanks". Ad: pizza delivery service.
-> {"show": false, "reason": "purchase already completed, ad arrives too late"}

Reply with ONLY the JSON object, no markdown fences: {"show": true/false, "reason": "<one short sentence>"}"""
