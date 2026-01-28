from tinygent.core.prompt import TinyPrompt


def get_new_clusters_proposal_prompt() -> TinyPrompt.UserSystem:
    return TinyPrompt.UserSystem(
        system=(
            'You are an expert knowledge-graph curator. '
            'Your task is to decide whether NEW CLUSTERS should be created. '
            'You are given existing clusters, candidate (similar) clusters, '
            'and the current entities under consideration. '
            'Avoid proposing duplicates or semantically equivalent clusters.'
        ),
        user="""<ENTITIES>
{% for e in entities %}
<ENTITY>
<ENTITY_NAME>{{ e.entity_name }}</ENTITY_NAME>
<ENTITY_SUMMARY>{{ e.entity_summary }}</ENTITY_SUMMARY>
</ENTITY>
{% endfor %}
</ENTITIES>

<EXISTING_CLUSTERS>
{% for cluster in existing_clusters %}
<CLUSTER>
<NAME>{{ cluster.name }}</NAME>
<SUMMARY>{{ cluster.summary }}</SUMMARY>

<ENTITIES>
{% for e in cluster.entities %}
<ENTITY>
<ENTITY_NAME>{{ e.entity_name }}</ENTITY_NAME>
<ENTITY_SUMMARY>{{ e.entity_summary }}</ENTITY_SUMMARY>
</ENTITY>
{% endfor %}
</ENTITIES>
</CLUSTER>
{% endfor %}
</EXISTING_CLUSTERS>

<CANDIDATE_CLUSTERS>
{% for cluster in candidate_clusters %}
<CLUSTER>
<NAME>{{ cluster.name }}</NAME>
<SUMMARY>{{ cluster.summary }}</SUMMARY>
</CLUSTER>
{% endfor %}
</CANDIDATE_CLUSTERS>

# TASK

Analyze the ENTITIES in context of:
- EXISTING_CLUSTERS (authoritative cluster assignments)
- CANDIDATE_CLUSTERS (semantically similar but not yet assigned)

Determine whether any IMPORTANT CLUSTER is missing.

Propose NEW CLUSTERS **only if**:
- The ENTITIES form a coherent theme not covered by EXISTING_CLUSTERS.
- The theme is not adequately represented by any CANDIDATE_CLUSTER.
- The cluster would meaningfully improve organization of the knowledge graph.

Do NOT propose a cluster if:
- Its meaning substantially overlaps with an EXISTING or CANDIDATE CLUSTER.
- It only differs by wording, scope narrowing, or trivial rephrasing.

# OUTPUT RULES

Return a JSON object matching this schema:

{
    "proposals": [
        {
            "name": string,
            "summary": string
        }
    ]
}

If no new clusters are needed, return:

{
    "proposals": []
}

# STYLE GUIDELINES

- Use concise, descriptive cluster NAMES.
- Summaries should explain the unifying concept in 1–2 sentences.
- Do NOT reference entities or clusters explicitly in the output.
- Do NOT restate existing cluster names.
""",
    )
