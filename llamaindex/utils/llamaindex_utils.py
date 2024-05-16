from typing import Any, Dict, List, Optional, Sequence, cast

from duvidas import clausulas
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.bridge.pydantic import Field
from llama_index.core.extractors import (
    BaseExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
)
from llama_index.core.extractors.interface import BaseExtractor

# assume documents are defined -> extract nodes
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType
from llama_index.core.settings import Settings

# ------------------------------ util classes ----------------------------------
DEFAULT_CLAUSE_EXTRACT_TEMPLATE = """\
Uma seção deste documento foi dedicada a responder à seguinte dúvida: ```{duvida}```.

Aqui está o conteúdo desta seção:
```{context_str}```

Identifica a cláusula desta seção entre as cláusulas abaixo:
{clausulas}

Clausula: """


# https://docs.llamaindex.ai/en/stable/module_guides/indexing/metadata_extraction/#custom-extractors
class CustomClauseExtractor(BaseExtractor):
    llm: LLMPredictorType = Field(description="The LLM to use for generation.")
    prompt_template: str = Field(
        default=DEFAULT_CLAUSE_EXTRACT_TEMPLATE,
        description="Template to use when identifying clauses.",
    )

    def __init__(
        self,
        llm: Optional[LLM] = None,
        llm_predictor: Optional[LLMPredictorType] = None,
        prompt_template: str = DEFAULT_CLAUSE_EXTRACT_TEMPLATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ):

        super().__init__(
            llm=llm or llm_predictor or Settings.llm,
            prompt_template=prompt_template,
            num_workers=num_workers,
            **kwargs,
        )

    async def _agenerate_node_summary(self, node: BaseNode) -> str:
        """Generate a summary for a node."""
        if self.is_text_node_only and not isinstance(node, TextNode):
            return ""

        context_str = node.get_content(metadata_mode=self.metadata_mode)
        summary = await self.llm.apredict(
            PromptTemplate(template=self.prompt_template),
            context_str=context_str,
            clausulas=clausulas,
        )

        return summary.strip()

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        if not all(isinstance(node, TextNode) for node in nodes):
            raise ValueError("Only `TextNode` is allowed for `Summary` extractor")

        node_summaries_jobs = []
        for node in nodes:
            node_summaries_jobs.append(self._agenerate_node_summary(node))

        node_summaries = await run_jobs(
            node_summaries_jobs,
            show_progress=self.show_progress,
            workers=self.num_workers,
        )

        # Extract node-level summary metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        for i, metadata in enumerate(metadata_list):
            if node_summaries[i]:
                metadata["section_clause"] = node_summaries[i]

        return metadata_list


# ------------------------------ util functions ----------------------------------

PT_QUESTION_GEN_TMPL = """\
Aqui está o contexto:
{context_str}

Dada a informação contextual, gere {num_questions} perguntas que este contexto pode fornecer respostas específicas \
que dificilmente seriam encontradas em outro lugar.

Resumos de nível mais alto do contexto circundante também podem ser fornecidos. Tente usar esses resumos para gerar melhores \
perguntas que este contexto possa responder.

"""

PT_SUMMARY_EXTRACT_TEMPLATE = """\
Aqui está o conteúdo da seção:
{context_str}

Resuma os principais tópicos e entidades da seção. \

Resumo:"""


def get_metadata_node_parser_pipeline():
    extractors = [
        # TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=3, prompt_template=PT_QUESTION_GEN_TMPL),
        # SummaryExtractor(
        #     # summaries=["prev", "self"],
        #     summaries=["self"],
        #     prompt_template=PT_SUMMARY_EXTRACT_TEMPLATE,
        # ),
        # CustomClauseExtractor(),
        # KeywordExtractor(keywords=10),
        # CustomExtractor()
    ]

    # Max chunk size for text-embedding-ada-002 is 2048 tokens
    NODE_PARSER_CHUNK_SIZE = 512
    NODE_PARSER_CHUNK_OVERLAP = 10

    sentence_splitter = SentenceSplitter.from_defaults(
        chunk_size=NODE_PARSER_CHUNK_SIZE, chunk_overlap=NODE_PARSER_CHUNK_OVERLAP
    )

    transformations = [sentence_splitter] + extractors

    pipeline = IngestionPipeline(transformations=transformations)
    return pipeline


prompt_template: str = DEFAULT_CLAUSE_EXTRACT_TEMPLATE


async def _agenerate_node_llm_answer(
    node: BaseNode, current_type: str, metadata_mode: MetadataMode = MetadataMode.NONE
) -> str:
    context_str = node.get_content(metadata_mode=metadata_mode)
    llm = Settings.llm
    result = await llm.apredict(
        PromptTemplate(template=prompt_template),
        context_str=context_str,
        clausulas=clausulas[current_type],
    )
    return result


async def _agenerate_all_nodes_llm_answer(
    nodes: List[BaseNode],
    current_type: str,
    duvida: str,
    metadata_mode: MetadataMode = MetadataMode.NONE,
) -> str:
    context_str = "\n\n".join(
        [node.get_content(metadata_mode=metadata_mode) for node in nodes]
    )
    llm = Settings.llm
    result = await llm.apredict(
        PromptTemplate(template=prompt_template),
        context_str=context_str,
        duvida=duvida,
        clausulas=clausulas[current_type],
    )
    return result


# ------------------------------ debugging helper functions ----------------------------------
def print_node_parsing_result(node, metadata_mode, only_metadata):
    print("\n\n----------------------- " + node.id_ + " -----------------------")
    if only_metadata:
        print(node.get_metadata_str(mode=metadata_mode))
    else:
        print(node.get_content(metadata_mode=metadata_mode))


def print_nodes_parsing_result(
    nodes, metadata_mode="all", only_metadata=False, ids_list=None
):
    print("length: ", len(nodes))
    for node in nodes:
        if ids_list is not None:
            # print(node.id_)
            if node.id_ in ids_list:
                print_node_parsing_result(node, metadata_mode, only_metadata)
        else:
            print_node_parsing_result(node, metadata_mode, only_metadata)
