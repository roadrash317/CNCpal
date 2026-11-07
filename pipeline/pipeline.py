import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import instructor
from llama_parse import LlamaParse
from openai import OpenAI
from pydantic import BaseModel, Field
from supabase import create_client, Client

# Load environment variables (.env file)
load_dotenv()

# =====================================================================
# 1. DEFINE SCHEMAS (Pydantic & JSONB Modifiers)
# =====================================================================

class CuttingParameters(BaseModel):
    vc_m_min: float = Field(description="Cutting speed (Vc) in meters per minute")
    fn_mm_rev: float = Field(description="Feed per revolution (fn) in mm/rev")
    ap_mm: Optional[float] = Field(default=None, description="Depth of cut (ap) in mm")


class CuttingToolSpec(BaseModel):
    brand: str = Field(description="Manufacturer brand, e.g., Widia, Iscar")
    catalog_number: str = Field(description="Exact catalog code, e.g., CNMG 432 RH WP35CT")
    insert_shape: str = Field(description="Insert shape geometry, e.g., CNMG")
    grade: str = Field(description="Insert grade designation, e.g., WP35CT, IC907")
    workpiece_material_group: str = Field(description="ISO material group or specific alloy name, e.g., 440C, M group")
    recommended_params: CuttingParameters
    
    # Dynamic JSONB field to capture unstructured catalog notes (e.g., chipbreakers)
    modifiers: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Flexible key-value storage for manufacturer-specific application limits or multipliers"
    )


class ToolCatalogExtraction(BaseModel):
    tools: List[CuttingToolSpec]


# =====================================================================
# 2. CORE PROCESSING SERVICE
# =====================================================================

class ToolIngestionPipeline:
    def __init__(self):
        # Initialize clients
        self.openai_client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL", ""), 
            os.getenv("SUPABASE_KEY", "")
        )
        
        # Configure LlamaParse using the mandatory v2 API layer
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",  # Preserves nested catalog data tables cleanly
            api_version="v2",        # <--- ADD THIS LINE HERE
            num_workers=4
        )
        )

    def extract_layout_from_pdf(self, file_path: str) -> str:
        """Stage 1: Extract structural text layout from PDF catalog pages."""
        print(f"[Stage 1] Parsing layout for {file_path}...")
        documents = self.parser.load_data(file_path)
        # Combine pages into a coherent markdown string
        return "\n\n".join([doc.text for doc in documents])

    def parse_with_llm(self, structured_text: str) -> ToolCatalogExtraction:
        """Stage 2: Standardize raw catalog markdown text into Pydantic models."""
        return self.openai_client.chat.completions.create(
            model="gpt-4o",
            response_model=ToolCatalogExtraction,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert production machining engineering system. "
                        "Accurately extract cutting tool specifications and speeds/feeds. "
                        "Populate the 'modifiers' dictionary with any specific application limits, "
                        "chipbreaker targets, machine constraints, or coolant instructions noted."
                    ),
                },
                {"role": "user", "content": f"Extract ALL cutting tools visible in this section of the catalog:\n\n{structured_text}"},
            ],
        )
    def save_to_supabase(self, extraction: ToolCatalogExtraction):
        """Stage 3: Verify and batch insert structured elements into PostgreSQL."""
        print(f"[Stage 3] Preparing batch insertion for {len(extraction.tools)} entries...")
        
        payload = []
        for tool in extraction.tools:
            # Structuring row array elements to match standard table schemas
            row = {
                "brand": tool.brand,
                "catalog_number": tool.catalog_number,
                "insert_shape": tool.insert_shape,
                "grade": tool.grade,
                "workpiece_material_group": tool.workpiece_material_group,
                "vc_m_min": tool.recommended_params.vc_m_min,
                "fn_mm_rev": tool.recommended_params.fn_mm_rev,
                "ap_mm": tool.recommended_params.ap_mm,
                "modifiers": tool.modifiers  # Maps directly to PostgreSQL JSONB column
            }
            payload.append(row)
            
        if not payload:
            print("No valid cutting data found to insert.")
            return

        # Execute upsert or batch insert
        try:
            response = self.supabase.table("cutting_tools").upsert(
                payload, 
                on_conflict="brand, catalog_number, workpiece_material_group"
            ).execute()
            print(f"Successfully processed batch ingestion. Rows inserted/updated: {len(response.data)}")
        except Exception as e:
            print(f"Database transaction failure: {str(e)}")


    def run_pipeline(self, pdf_path: str):
        """Orchestrate entire pipeline execution using text splitting."""
        # 1. Extract raw layout text
        markdown_text = self.extract_layout_from_pdf(pdf_path)
        
        # 2. Break the catalog text into manageable pieces (roughly by section/page)
        print("[System] Splitting catalog text into chunks for deeper extraction...")
        chunks = [markdown_text[i:i+4000] for i in range(0, len(markdown_text), 4000)]
        
        # 3. Process each piece individually and accumulate the results
        for idx, chunk in enumerate(chunks):
            print(f"--> Processing text chunk {idx + 1} of {len(chunks)}...")
            try:
                structured_data = self.parse_with_llm(chunk)
                if structured_data.tools:
                    self.save_to_supabase(structured_data)
            except Exception as e:
                print(f"Skipped chunk {idx + 1} due to parsing limit: {str(e)}")


# =====================================================================
# 3. EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    pipeline = ToolIngestionPipeline()
    
    # Path to your downloaded cutting tool manufacturer PDF catalog sheet
    catalog_pdf = "widia_turning_catalog.pdf" 
    
    if os.path.exists(catalog_pdf):
        pipeline.run_pipeline(catalog_pdf)
    else:
        print(f"Place your tool document at '{catalog_pdf}' to process ingestion.")
