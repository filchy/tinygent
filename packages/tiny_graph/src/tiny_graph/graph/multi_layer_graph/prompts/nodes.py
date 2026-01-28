from tinygent.core.prompt import TinyPrompt


def get_llm_resolve_duplicites_prompt_template() -> TinyPrompt.UserSystem:
    return TinyPrompt.UserSystem(
        system='You are a helpful assistant that determines whether or not ENTITIES extracted from a conversation are duplicates of existing entities.',
        user="""<PREVIOUS MESSAGES>
{{ previous_events }}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{{ current_event }}
</CURRENT MESSAGE>


Each of the following ENTITIES were extracted from the CURRENT MESSAGE.
Each entity in ENTITIES is represented as a JSON object with the following structure:
{
    id: integer id of the entity,
    name: "name of the entity",
    entity_type: ["Entity", "<optional additional label>", ...],
    entity_type_description: "Description of what the entity type represents"
}

<ENTITIES>
{{ extracted_entities }}
</ENTITIES>

<EXISTING ENTITIES>
{{ existing_entities }}
</EXISTING ENTITIES>

Each entry in EXISTING ENTITIES is an object with the following structure:
{
    idx: integer index of the candidate entity (use this when referencing a duplicate),
    name: "name of the candidate entity",
    entity_types: ["Entity", "<optional additional label>", ...],
    ...<additional attributes such as summaries or metadata>
}

For each of the above ENTITIES, determine if the entity is a duplicate of any of the EXISTING ENTITIES.

Entities should only be considered duplicates if they refer to the *same real-world object or concept*.

Do NOT mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.

Task:
ENTITIES contains {{ extracted_entities|length }} entities with IDs 0 through {{ (extracted_entities|length) - 1 }}.
Your response MUST include EXACTLY {{ extracted_entities|length }} resolutions with IDs 0 through {{ (extracted_entities|length) - 1}}. Do not skip or add IDs.

For every entity, return an object with the following keys:
{
    "id": integer id from ENTITIES,
    "name": the best full name for the entity (preserve the original name unless a duplicate has a more complete name),
    "duplicate_idx": the idx of the EXISTING ENTITY that is the best duplicate match, or -1 if there is no duplicate,
    "duplicates": a sorted list of all idx values from EXISTING ENTITIES that refer to duplicates (deduplicate the list, use [] when none or unsure)
}

- Only use idx values that appear in EXISTING ENTITIES.
- Set duplicate_idx to the smallest idx you collected for that entity, or -1 if duplicates is empty.
- Never fabricate entities or indices.""",
    )


def get_entity_attributes_extraction_prompt() -> TinyPrompt.UserSystem:
    return TinyPrompt.UserSystem(
        system='You are a helpful assistant that extracts entity properties from the provided text.',
        user="""Given the MESSAGES and the following ENTITY, update any of its attributes based on the information provided
in MESSAGES. Use the provided attribute descriptions to better understand how each attribute should be determined.

Guidelines:
1. Do not hallucinate entity property values if they cannot be found in the current context.
2. Only use the provided MESSAGES and ENTITY to set attribute values.

<MESSAGES>
{{ [previous_events | join(', ') if previous_events else None, event_content]
   | select
   | join('\n') }}
</MESSAGES>

<ENTITY>
{{ entity }}
</ENTITY>""",
    )


def get_entity_summary_creation_prompt() -> TinyPrompt.UserSystem:
    return TinyPrompt.UserSystem(
        system='You are a helpful assistant that extracts entity summaries from the provided text.',
        user="""Given the MESSAGES and the ENTITY, update the summary that combines relevant information about the entity
from the messages and relevant information from the existing summary. Write only new summarization text, nothing else.

<EXISTING SUMMARY>
{{ existing_summary }}
</EXISTING SUMMARY>

<MESSAGES>
{{ [previous_events | join(', ') if previous_events else None, event_content]
   | select
   | join('\n') }}
</MESSAGES>

<ENTITY>
{{ entity }}
</ENTITY>""",
    )
