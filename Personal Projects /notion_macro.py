# import re
# from notion_client import Client

# # Config
# NOTION_TOKEN = "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"
# PAGE_ID = "2cfedc08c89580879ffecd5c367b1e24"

# notion = Client(auth=NOTION_TOKEN)

# def convert_text_to_rich_text(text):
#     # This regex handles multi-line math and text-wrapped math more reliably
#     pattern = r'(\$\$.*?\$\$|\$.*?\$)'
#     parts = re.split(pattern, text, flags=re.DOTALL)
#     rich_text = []

#     for part in parts:
#         if not part: continue
        
#         if (part.startswith('$$') and part.endswith('$$')) or (part.startswith('$') and part.endswith('$')):
#             math_content = part.strip('$').strip()
#             rich_text.append({"type": "equation", "equation": {"expression": math_content}})
#         else:
#             rich_text.append({"type": "text", "text": {"content": part}})
    
#     return rich_text

# def process_blocks(block_id):
#     blocks = notion.blocks.children.list(block_id=block_id).get("results")
    
#     for block in blocks:
#         block_type = block["type"]
        
#         # Check if this block type has text (Paragraph, Callout, Quote, Bulleted List, etc.)
#         if "rich_text" in block[block_type]:
#             rich_text_array = block[block_type]["rich_text"]
#             text_content = "".join([t["plain_text"] for t in rich_text_array])
            
#             if "$" in text_content:
#                 new_rich_text = convert_text_to_rich_text(text_content)
#                 # Update the specific block type
#                 notion.blocks.update(block_id=block["id"], **{block_type: {"rich_text": new_rich_text}})
#                 print(f"Updated {block_type}: {text_content[:30]}...")

#         # If the block has children (like a Toggle or a Column), process those too
#         if block.get("has_children"):
#             process_blocks(block["id"])

# if __name__ == "__main__":
#     print("Starting deep scan...")
#     process_blocks(PAGE_ID)
#     print("Done!")

import re
import sys
import time
from notion_client import Client

# REPLACING THIS WITH YOUR SECRET TOKEN IS ESSENTIAL
NOTION_TOKEN = "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"
notion = Client(auth=NOTION_TOKEN)

def extract_id(url_or_id):
    match = re.search(r'([a-f0-9]{32})', url_or_id)
    return match.group(1) if match else url_or_id

def convert_text_to_rich_text(text):
    pattern = r'(\$\$.*?\$\$|\$.*?\$)'
    parts = re.split(pattern, text, flags=re.DOTALL)
    rich_text = []
    for part in parts:
        if not part: continue
        if (part.startswith('$$') and part.endswith('$$')) or (part.startswith('$') and part.endswith('$')):
            math_content = part.strip('$').strip()
            if math_content:
                rich_text.append({"type": "equation", "equation": {"expression": math_content}})
            else:
                rich_text.append({"type": "text", "text": {"content": part}})
        else:
            rich_text.append({"type": "text", "text": {"content": part}})
    return rich_text

def process_blocks(block_id):
    try:
        has_more = True
        start_cursor = None
        while has_more:
            response = notion.blocks.children.list(block_id=block_id, start_cursor=start_cursor)
            results = response.get("results")
            
            for block in results:
                block_type = block["type"]
                
                # --- CASE 1: Standard Blocks (Paragraphs, Callouts, Lists) ---
                if "rich_text" in block.get(block_type, {}):
                    full_text = "".join([t.get("plain_text", "") for t in block[block_type]["rich_text"]])
                    if "$" in full_text:
                        new_rich_text = convert_text_to_rich_text(full_text)
                        notion.blocks.update(block_id=block["id"], **{block_type: {"rich_text": new_rich_text}})
                        print(f"Fixed {block_type}")

                # --- CASE 2: Tables (Rows and Cells) ---
                elif block_type == "table_row":
                    cells = block["table_row"]["cells"]
                    updated_cells = []
                    needs_update = False
                    
                    for cell in cells:
                        cell_text = "".join([t.get("plain_text", "") for t in cell])
                        if "$" in cell_text:
                            updated_cells.append(convert_text_to_rich_text(cell_text))
                            needs_update = True
                        else:
                            updated_cells.append(cell)
                    
                    if needs_update:
                        notion.blocks.update(block_id=block["id"], table_row={"cells": updated_cells})
                        print("Fixed Table Row")

                # Recurse into children (Important for Tables because they are parent blocks)
                if block.get("has_children"):
                    process_blocks(block["id"])
            
            has_more = response.get("has_more")
            start_cursor = response.get("next_cursor")
            time.sleep(0.05) # Tiny buffer for API stability

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_id = extract_id(sys.argv[1])
        print(f"Deep Scanning Page: {target_id}")
        process_blocks(target_id)
        print("--- All Done! ---")
    else:
        print("Error: No input provided.")