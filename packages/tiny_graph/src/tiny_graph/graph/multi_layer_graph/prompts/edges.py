from tinygent.core.prompt import TinyPrompt


def get_cluster_edge_extraction_prompt() -> TinyPrompt.UserSystem:
    return TinyPrompt.UserSystem(
        system='You are an expert knowledge graph extraction engine. '
        'You extract semantic relationships between CLUSTERS and ENTITIES '
        'based strictly on the provided event content.',
        user="""<CURRENT_MESSAGE>
{{ event_content }}
</CURRENT_MESSAGE>

<ENTITIES>
{{ entities }}
</ENTITIES>

<CLUSTERS>
{{ clusters }}
</CLUSTERS>

# TASK
Extract all valid relationships where a CLUSTER is related to an ENTITY based on the CURRENT MESSAGE.

Only extract edges that:
- connect ONE CLUSTER to ONE ENTITY,
- are explicitly stated or unambiguously implied by the CURRENT MESSAGE,
- represent meaningful semantic grouping, categorization, membership, association, or contextual linkage.

{{ custom_prompt }}

# OUTPUT FORMAT
Return a JSON object that conforms EXACTLY to this schema:

{
  "edges": [
    {
      "source_cluster_id": <int>,   // from CLUSTERS.id
      "target_entity_id": <int>     // from ENTITIES.id
    }
  ]
}

# EXTRACTION RULES

1. **ID STRICTNESS**
   - `source_cluster_id` MUST come from CLUSTERS.id
   - `target_entity_id` MUST come from ENTITIES.id
   - Any ID not in the lists is INVALID

2. Do NOT invent clusters or entities.

3. Do NOT output duplicate edges.

4. If no valid relationships exist, return:
   { "edges": [] }

5. Do NOT include explanations, comments, metadata, or additional fields.
   - Output JSON only.
        """,
    )


def get_cluster_edge_extract_reflextion_prompt() -> TinyPrompt.UserSystem:
    return TinyPrompt.UserSystem(
        system='',
        user='',
    )


def get_entity_edge_extraction_prompt() -> TinyPrompt.UserSystem:
    return TinyPrompt.UserSystem(
        system='You are an expert fact extractor that extracts fact triples from text. '
        '1. Extracted fact triples should also be extracted with relevant date information.'
        '2. Treat the CURRENT TIME as the time the CURRENT MESSAGE was sent. All temporal information should be extracted relative to this time.',
        user="""<CURRENT_DATE>
{{ current_date }}
</CURRENT_DATE>

<FACT TYPES>
{{ edge_types }}
</FACT TYPES>

<PREVIOUS_MESSAGES>
{{ previous_events }}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{{ event_content }}
</CURRENT_MESSAGE>

<ENTITIES>
{{ entities }}
</ENTITIES>

<REFERENCE_TIME>
{context['reference_time']}  # ISO 8601 (UTC); used to resolve relative time mentions
</REFERENCE_TIME>

# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
    and can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.
- The FACT TYPES provide a list of the most important types of facts, make sure to extract facts of these types
- The FACT TYPES are not an exhaustive list, extract all facts from the message even if they do not fit into one
    of the FACT TYPES
- The FACT TYPES each contain their fact_type_signature which represents the source and target entity types.

You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.

{{ custom_prompt }}

# EXTRACTION RULES

1. **Entity ID Validation**: `source_entity_id` and `target_entity_id` must use only the `id` values from the ENTITIES list provided above.
   - **CRITICAL**: Using IDs not in the list will cause the edge to be rejected
2. Each fact must involve two **distinct** entities.
3. Use a SCREAMING_SNAKE_CASE string as the `relation_type` (e.g., FOUNDED, WORKS_AT).
4. Do not emit duplicate or semantically redundant facts.
5. The `fact` should closely paraphrase the original source sentence(s). Do not verbatim quote the original text.
6. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions (e.g., "last week").
7. Do **not** hallucinate or infer temporal bounds from unrelated events.

# DATETIME RULES

- Use ISO 8601 with “Z” suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set `valid_at` to REFERENCE_TIME.
- If a change/termination is expressed, set `invalid_at` to the relevant timestamp.
- Leave both fields `null` if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.""",
    )


def get_entity_edge_extract_reflextion_prompt() -> TinyPrompt.UserSystem:
    return TinyPrompt.UserSystem(
        system='You are an AI assistant that determines which facts have not been extracted from the given context',
        user="""<PREVIOUS MESSAGES>
{{ previous_events }}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{{ event_content }}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{{ entities }}
</EXTRACTED ENTITIES>

<EXTRACTED FACTS>
{{ extracted_facts }}
</EXTRACTED FACTS>

Given the above MESSAGES, list of EXTRACTED ENTITIES entities, and list of EXTRACTED FACTS; 
determine if any facts haven't been extracted.""",
    )


def get_resolve_edge_prompt() -> TinyPrompt.UserSystem:
    return TinyPrompt.UserSystem(
        system='You are a helpful assistant that de-duplicates facts from fact lists and determines which existing facts are contradicted by the new fact.',
        user="""Task:
You will receive TWO separate lists of facts. Each list uses 'idx' as its index field, starting from 0.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - Facts with similar information that contain key differences should NOT be marked as duplicates.
   - Return idx values from EXISTING FACTS.
   - If no duplicates, return an empty list for duplicate_facts.

2. FACT TYPE CLASSIFICATION:
   - Given the predefined FACT TYPES, determine if the NEW FACT should be classified as one of these types.
   - Return the fact type as fact_type or DEFAULT if NEW FACT is not one of the FACT TYPES.

3. CONTRADICTION DETECTION:
   - Based on FACT INVALIDATION CANDIDATES and NEW FACT, determine which facts the new fact contradicts.
   - Return idx values from FACT INVALIDATION CANDIDATES.
   - If no contradictions, return an empty list for contradicted_facts.

IMPORTANT:
- duplicate_facts: Use ONLY 'idx' values from EXISTING FACTS
- contradicted_facts: Use ONLY 'idx' values from FACT INVALIDATION CANDIDATES
- These are two separate lists with independent idx ranges starting from 0

Guidelines:
1. Some facts may be very similar but will have key differences, particularly around numeric values in the facts.
    Do not mark these facts as duplicates.

<FACT TYPES>
{{ edge_types }}
</FACT TYPES>

<EXISTING FACTS>
{{ existing_edges }}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{{ edge_invalidation_candidates }}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{{ new_edge }}
</NEW FACT>""",
    )


def get_extract_edge_attributes() -> TinyPrompt.UserSystem:
    return TinyPrompt.UserSystem(
        system='You are a helpful assistant that extracts fact properties from the provided text.',
        user="""<MESSAGE>
{{ event_content }}
</MESSAGE>
<REFERENCE TIME>
{{ reference_time }}
</REFERENCE TIME>

Given the above MESSAGE, its REFERENCE TIME, and the following FACT, update any of its attributes based on the information provided
in MESSAGE. Use the provided attribute descriptions to better understand how each attribute should be determined.

Guidelines:
1. Do not hallucinate entity property values if they cannot be found in the current context.
2. Only use the provided MESSAGES and FACT to set attribute values.

<FACT>
{{ fact }}
</FACT>""",
    )
