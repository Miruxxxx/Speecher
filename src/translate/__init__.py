"""Live translation of the transcript (docs/ROADMAP.md stage 3).

Deliberately not part of `src/llm`: translation does not go through LM Studio.
It is a local seq2seq model in its own thread, so it never competes with the
answer/summary queue the buttons use — pressing "Ответить" must not wait behind
a stream of background translations.
"""
