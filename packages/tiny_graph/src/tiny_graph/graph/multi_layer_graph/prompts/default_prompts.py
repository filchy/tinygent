from tiny_graph.graph.multi_layer_graph.multi_layer_graph import (
    EntityExtractorPromptTemplate,
)
from tiny_graph.graph.multi_layer_graph.multi_layer_graph import (
    TinyMultiLayerGraphTemplate,
)
from tinygent.core.prompt import TinyPrompt


def get_prompt_template() -> TinyMultiLayerGraphTemplate:
    return TinyMultiLayerGraphTemplate(
        entity_extractor=EntityExtractorPromptTemplate(
            extract_text=TinyPrompt.UserSystem(
                system='You are an AI assistant that extracts entity nodes from text. Your primary task is to extract and classify the speaker and other significant entities mentioned in the provided text.',
                user="""<ENTITY TYPES>
{{ entity_types }}
</ENTITY TYPES>

<TEXT>
{{ event_content }}
</TEXT>

Given the above text, extract entities from the TEXT that are explicitly or implicitly mentioned.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
Indicate the classified entity type by providing its entity_type_id.

{{ custom_prompt }}

Guidelines:
1. Extract significant entities, concepts, or actors mentioned in the conversation.
2. Avoid creating nodes for relationships or actions.
3. Avoid creating nodes for temporal information like dates, times or years (these will be added to edges later).
4. Be as explicit as possible in your node names, using full names and avoiding abbreviations.
                """,
            ),
            extract_message=TinyPrompt.UserSystem(
                system='You are an AI assistant that extracts entity nodes from conversational messages. Your primary task is to extract and classify the speaker and other significant entities mentioned in the conversation.',
                user="""<ENTITY TYPES>
{{ entity_types }}
</ENTITY TYPES>

<PREVIOUS MESSAGES>
[{{ previous_events | join(', ') }}]
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{{ event_content }}
</CURRENT MESSAGE>

Instructions:

You are given a conversation context and a CURRENT MESSAGE. Your task is to extract **entity nodes** mentioned **explicitly or implicitly** in the CURRENT MESSAGE.
Pronoun references such as he/she/they or this/that/those should be disambiguated to the names of the 
reference entities. Only extract distinct entities from the CURRENT MESSAGE. Don't extract pronouns like you, me, he/she/they, we/us as entities.

1. **Speaker Extraction**: Always extract the speaker (the part before the colon `:` in each dialogue line) as the first entity node.
   - If the speaker is mentioned again in the message, treat both mentions as a **single entity**.

2. **Entity Identification**:
   - Extract all significant entities, concepts, or actors that are **explicitly or implicitly** mentioned in the CURRENT MESSAGE.
   - **Exclude** entities mentioned only in the PREVIOUS MESSAGES (they are for context only).

3. **Entity Classification**:
   - Use the descriptions in ENTITY TYPES to classify each extracted entity.
   - Assign the appropriate `entity_type_id` for each one.

4. **Exclusions**:
   - Do NOT extract entities representing relationships or actions.
   - Do NOT extract dates, times, or other temporal information—these will be handled separately.

5. **Formatting**:
   - Be **explicit and unambiguous** in naming entities (e.g., use full names when available).

{{ custom_prompt }}
                """,
            ),
            extract_json=TinyPrompt.UserSystem(
                system='You are an AI assistant that extracts entity nodes from JSON.  Your primary task is to extract and classify relevant entities from JSON files',
                user="""<ENTITY TYPES>
{{ entity_types }}
</ENTITY TYPES>

<SOURCE DESCRIPTION>:
{{ source_description }}
</SOURCE DESCRIPTION>
<JSON>
{{ event_content }}
</JSON>

{{ custom_prompt }}

Given the above source description and JSON, extract relevant entities from the provided JSON.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
Indicate the classified entity type by providing its entity_type_id.

Guidelines:
1. Extract all entities that the JSON represents. This will often be something like a "name" or "user" field
2. Extract all entities mentioned in all other properties throughout the JSON structure
3. Do NOT extract any properties that contain dates
                """,
            ),
            reflexion=TinyPrompt.UserSystem(
                system='You are an AI assistant that determines which entities have not been extracted from the given context',
                user="""<PREVIOUS MESSAGES>
[{{ previous_events | join(', ') }}]
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{{ event_content }}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
[{{ extracted_entities | join(', ') }}]
</EXTRACTED ENTITIES>

Given the above previous messages, current message, and list of extracted entities; determine if any entities haven't been
extracted.
                """,
            ),
        )
    )
