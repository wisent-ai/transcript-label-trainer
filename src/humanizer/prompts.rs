//! The instructions the humanizer's teacher, reviewer, student and judge read.
//! `SYSTEM_PROMPT` is the student's own prompt: its SHA-256 is recorded in
//! `preparation.json`, so changing a byte of it is a new dataset.

pub(crate) const SYSTEM_PROMPT: &str = "Faithfully paraphrase the supplied text. Preserve its \
complete meaning, intent, level of certainty, facts, names, product and company names, technical \
terms, numbers, dates, amounts, URLs, email addresses, quotations, code, commands, meaningful \
formatting, emotional force, and calls to action. Keep the source language and register. Do not \
add claims, examples, greetings, conclusions, or context. Return only the rewritten text, with no \
analysis, label, preface, or notes.";

pub(crate) const TEACHER_PROMPT: &str = "Create the source side of an inverse style-transfer \
example. Rewrite the authored message as generic polished AI-assistant prose in the same language. \
Preserve every fact, name, technical term, number, date, amount, URL, email address, quotation, \
command, intent, uncertainty, emotional force, and call to action. Do not summarize, answer, \
explain, censor, or add information. Make the wording and rhythm substantially more generic and \
AI-like. Return only the rewritten message.";

pub(crate) const REVIEW_PROMPT: &str =
    "Judge one proposed inverse style-transfer pair. The source \
must be generic polished AI prose; the target must preserve the same complete meaning while \
retaining the author's natural voice. Return exactly JSON with booleans faithful, generic_ai, \
same_language, and usable. usable may be true only when all other fields are true and neither \
side adds or drops any fact, name, number, technical term, request, question, uncertainty, or \
emotional force.";

pub(crate) const JUDGE_PROMPT: &str = "Compare a base model and a trained personal-voice model on \
one held-out style-transfer case. The source is generic AI prose. The reference is a real message \
by the target author. Score each candidate independently from 0 to 1 for semantic_fidelity to the \
source and voice_match to the reference author's cadence, directness, register, and phrasing \
without requiring exact wording. ai_boilerplate is true when canned AI phrasing remains. passed \
is true only when semantic_fidelity is at least 0.95, voice_match is at least 0.75, and \
ai_boilerplate is false. Return exactly JSON: \
{\"base\":{\"semantic_fidelity\":0.0,\"voice_match\":0.0,\"ai_boilerplate\":true,\"passed\":false},\
\"student\":{\"semantic_fidelity\":0.0,\"voice_match\":0.0,\"ai_boilerplate\":true,\"passed\":false}}.";
