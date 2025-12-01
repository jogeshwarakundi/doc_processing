# Class ExtractionAgent
# Function run_extraction_agent()
# Take the model information (nomic-embed-text + gpt-5-nano as defaults) and input file (docx) as input
# Utilize the docx_reader to read the input file information
# For each section, first look up the section header info as a chunk and figure out the strategy (paragraph vs section vs table)
#if strategy is paragraph, then for each paragraph in the section, look up the paragraph text as a chunk
#if strategy is section, then look up the entire section text as a chunk
# mark the table as TODO for now
# For each chunk, use the prompt template to generate the prompt
# fire prompt and get the JSON response using the model (default is gpt-5-nano)
# Add all the JSOn responses into a list
# Finally use the utils.py to merge all the JSON fragments into a single JSON object
# return that JSON output

import argparse
import os
import json
import logging
from typing import List

from docx_reader import DocxReader
from db_handler import DBHandler
from utils import merge_json_fragments
import openai


class ExtractionAgent:
    def __init__(self, chroma_dir: str = "db", embedding_model: str = "nomic-embed-text", llm_model: str = "gpt-5-nano"):
        self.handler = DBHandler(chroma_dir)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        # Class-specific logger
        self.logger = logging.getLogger(self.__class__.__name__)
        # If no handler configured, default to WARNING level and allow propagation
        if not self.logger.handlers:
            # Don't add handlers here; let application configure logging. Set default level.
            self.logger.setLevel(logging.INFO)

    def _determine_strategy(self, section_title: str) -> str:
        """Query the DB using the section title to determine parsing strategy.

        Expected stored metadata currently uses the key 'prompt' (the preloader stored
        prompt or parsing_strategy under that key). We treat values 'paragraph' or
        'section' or 'table' (case-insensitive) as explicit signals. Defaults to
        'paragraph'.
        """
        try:
            hits = self.handler.query(section_title, self.embedding_model, top_k=1, distance_threshold=0.1)
        except Exception:
            return "paragraph"

        if not hits:
            return "paragraph"

        meta = hits[0].get("metadata") or {}
        # The preloader stored its value under 'prompt' key
        val = meta.get("prompt") if isinstance(meta, dict) else None
        if not val:
            return "paragraph"

        sval = str(val).strip().lower()
        if "ignore" in sval:
            return "ignore"
        if "section" in sval:
            return "section"
        if "table" in sval:
            return "table"
        if "paragraph" in sval:
            return "paragraph"

        # If value is something else, default to paragraph
        return "paragraph"

    def _build_prompt(self, chunk_text: str) -> str:
        """Given a chunk of text, look up the top DB hit and extract the stored prompt.

        Behavior:
        - Run a vector query for the chunk_text using the embedding model configured on
          this agent.
        - Take the first hit (if any) and look for a 'prompt' entry in its metadata.
        - If no prompt is found, raise ValueError.
        - Otherwise append the chunk_text to the prompt template and return the full
          prompt string.
        """
        try:
            hits = self.handler.query(chunk_text, self.embedding_model, top_k=1, distance_threshold=0.1)
        except Exception as e:
            raise RuntimeError(f"Failed to query DB for prompt template: {e}")

        if not hits:
            raise ValueError("No DB hits found for chunk; cannot determine prompt template.")

        top = hits[0]
        metadata = top.get("metadata") or {}

        # The preloader stored the prompt or parsing_strategy under the 'prompt' key.
        prompt_template = None
        if isinstance(metadata, dict):
            prompt_template = metadata.get("prompt")

        if not prompt_template or not isinstance(prompt_template, str):
            raise ValueError("Top DB hit does not contain a usable 'prompt' in metadata.")

        # Combine the stored template and the actual chunk text.
        full_prompt = f"{prompt_template}\n{chunk_text}"
        return full_prompt

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI Responses API and return text output (expected JSON string).

        Uses `openai.responses.create(...)` per the Responses API migration.
        """
        # Ensure API key is present via environment if needed
        if not os.environ.get("OPENAI_API_KEY"):
            self.logger.warning("OPENAI_API_KEY not set; LLM call will likely fail.")

        try:
            # Responses API: pass the prompt as `input`.
            resp = openai.responses.create(
                model=self.llm_model,
                input=prompt
            )

            if getattr(resp, "error", None):
                self.logger.error(f"_call_llm: ERROR: LLM call returned error: {getattr(resp, 'error')}")
                return "{}"
            return getattr(resp, "output_text", str(resp))
        except Exception as e:
            self.logger.error(f"_call_llm: ERROR: LLM call failed: {e}")
            return "{}"


    def process_text(self, input_text: str) -> str:
        #First build the prompt for this paragraph
        self.logger.info("--------------------------------")
        self.logger.info(f"ProcessText: Inside: {input_text}")
        if not input_text or not input_text.strip():
            self.logger.info(f"ProcessText: Return: empty")
            self.logger.info("--------------------------------")
            return "{}"
        try:
            prompt = self._build_prompt(input_text)
        except Exception as e:
            self.logger.error(f"ProcessText: ERROR: Failed to build prompt: {e}")
            self.logger.info("--------------------------------")
            return "{}"

        #Else take the prompt and call the LLM
        self.logger.info(f"ProcessText: Invoke LLM with prompt:\n{prompt}")
        response = self._call_llm(prompt)
        self.logger.info(f"ProcessText: Return: LLM response\n{response}")
        self.logger.info("--------------------------------")
        return response

    def run_extraction_agent(self, docx_path: str) -> str:
        reader = DocxReader(docx_path)
        sections = reader.parse()

        fragments: List[str] = []

        for sec in sections:
            title = sec.get("title", "")
            strategy = self._determine_strategy(title)
            self.logger.info(f"Processing section '{title}' with strategy: {strategy}")
            
            if strategy == "ignore":
                self.logger.info(f"Ignoring section '{title}' as per strategy.")
                continue
            
            if strategy not in ["paragraph", "section", "table"]:
                self.logger.warning(f"Unknown strategy '{strategy}' for section '{title}'.")
                #throw error here. This is invalid input.
                raise ValueError(f"Invalid strategy '{strategy}' for section '{title}'.")

            if strategy == "paragraph":
                for p in sec.get("paragraphs", []):
                    text = p.get("text", "")
                    resp = self.process_text(text)
                    fragments.append(resp)

            elif strategy == "section":
                # create the full section text by accumulating all paragraph texts
                section_texts = []
                for p in sec.get("paragraphs", []):
                    text = p.get("text", "")
                    if text:
                        section_texts.append(text)
                full_section_text = "\n".join(section_texts)
                if full_section_text:
                    resp = self.process_text(full_section_text)
                    fragments.append(resp)
                else:
                    self.logger.warning(f"Section '{title}' has no paragraph text to process.")

            elif strategy == "table":
                # For table strategy: collect any JSON fragments found by querying
                # paragraph text against the DB, and include the section's tables.
                self.logger.info(f"Processing tables for section '{title}'")

                paragraph_jsons = []
                for p in sec.get("paragraphs", []):
                    p_text = p.get("text", "")
                    resp = self.process_text(p_text)
                    paragraph_jsons.append(resp)

                # TODO: We need to process these tables too
                # check what comes in their table_to_dict() stuff
                # How do we return these tables?
                tables = sec.get("tables", [])

                chunk_obj = {
                    "paragraphs": paragraph_jsons,
                    "tables": tables,
                }

                fragments.append(json.dumps(chunk_obj))

            else:
                # default to paragraph processing
                for p in sec.get("paragraphs", []):
                    text = p.get("text", "")
                    prompt = self._build_prompt(text)
                    resp = self._call_llm(prompt)
                    fragments.append(resp)

        merged = merge_json_fragments(fragments)
        return merged


def main():
    parser = argparse.ArgumentParser(description="Run extraction agent on a DOCX file.")
    parser.add_argument("--docx", required=True, help="Path to the DOCX file to process.")
    parser.add_argument("--chroma-dir", default="db", help="Chroma DB directory to use.")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="Embedding model used for vector lookup (default: nomic-embed-text for local Ollama).")
    parser.add_argument("--llm-model", default="gpt-5-nano", help="LLM model used for extraction prompts.")
    parser.add_argument("--output", help="Path to write the merged JSON output (overwrites if exists). If omitted, output is printed only.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO")

    args = parser.parse_args()

    # Configure logging according to CLI
    import logging as _logging
    try:
        lvl = getattr(_logging, args.log_level.upper())
    except Exception:
        lvl = _logging.INFO
    _logging.basicConfig(level=lvl, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    agent = ExtractionAgent(chroma_dir=args.chroma_dir, embedding_model=args.embedding_model, llm_model=args.llm_model)
    if args.embedding_model != "nomic-embed-text":
        print(f"[INFO] Using embedding model: {args.embedding_model}")
    output = agent.run_extraction_agent(args.docx)
    print("\n=== Merged JSON Output ===")
    print(output)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"[INFO] Wrote merged JSON to {args.output}")
        except Exception as e:
            print(f"[ERROR] Failed to write output file {args.output}: {e}")


if __name__ == "__main__":
    main()
