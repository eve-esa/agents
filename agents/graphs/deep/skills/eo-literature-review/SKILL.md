---
name: eo-literature-review
description: Write a sourced literature review on an Earth Observation topic from EVE's document collection. Use when the user asks for a review, state of the art, survey, or an overview of what the literature says on a topic, mission, sensor, or method.
---

# Earth Observation literature review

Use this skill only when eve_retrieval_retrieve is in your tool list. If it is not, say that document retrieval is not enabled for this conversation and give a short overview from general knowledge, marked as such.

## Steps

1. Split the topic into three to five sub-questions, for example: definition and principles, data and missions, methods, applications, open challenges. Write them as headings in /review_notes.md with write_file.
2. For each sub-question, call eve_retrieval_retrieve with a concise, standalone query. Keep the requested dates and the selected collection scope.
3. After each call, add the useful findings under the matching heading in /review_notes.md with edit_file: one bullet per finding, with the title of the source it came from. Skip results that are irrelevant.
4. If a sub-question returns nothing useful, reformulate the query once. If it still returns nothing, note the gap and move on.
5. Read /review_notes.md and write the review.

## Review format

- A two or three sentence summary of the state of the field.
- One heading per sub-question, with bullet points grounded in the retrieved sources.
- A "Gaps and open questions" heading covering what the sources disagree on or do not cover.
- A "Sources" list with only the documents that were actually returned.

Never cite a document that eve_retrieval_retrieve did not return.
