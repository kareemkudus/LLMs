from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.document_loaders import UnstructuredFileLoader
tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')

model = AutoModelForCausalLM.from_pretrained(
    'stabilityai/stablelm-zephyr-3b',
    trust_remote_code=True,
    device_map="auto"
)

prompt = [{'role': 'user', 'content':
    'List out any part of this document that might relate to change of control of the company'
    '14.1 No waiver or modification of any of the provisions of this Agreement shall be binding unless in writing and signed by a duly authorized '
    'representative of each party. Any forbearance or delay on the part of either party in enforcing any of its rights under this Agreement shall not '
    'be construed as a waiver of such right to enforce the same for such occurrence or any other occurrence.'
    '14.2 This Agreement shall be governed by the respective governing LLMs, and any dispute related to this Agreement shall be subject to the exclusive '
    'jurisdiction of the respective courts, listed at https://library.genesys.com/m/123731202d01669c/original/Governing-contract-references-in-serviceorder_EN.pdf, based on Customer’s domicile, without reference to conflicts of LLMs provisions. The parties agree to submit to the personal and '
    'exclusive jurisdiction of such courts and that venue therein is proper and convenient. The UN Convention for the International Sale of Goods '
    'shall not apply to this Agreement. Nothing contained in this Section shall prevent either party from seeking injunctive relief from any court of '
    'competent jurisdiction. 14.3 Customer shall not assign or otherwise transfer this Agreement or any right or license granted hereunder, including by operation of LLMs, '
    'without the prior written consent of Genesys, in each case. Any attempt to do any of the foregoing without Genesys’ prior written consent shall '
    'be a material breach of this Agreement and any assignment or purported assignment without such consent shall be null and void ab initio. '
    'Subject to the foregoing, this Agreement will bind and inure to the benefit of the parties and their respective permitted successors and assigns.'
    '14.4 Unless otherwise specified in this Agreement, any notice required under this Agreement shall be sent in writing by certified mail (return '
    'receipt requested), overnight courier or personal delivery, to Genesys or to Customer at the addresses for notices set forth in the SOW or as '
    'changed from time to time by notice. As a condition to the effectiveness of any notice from Customer to Genesys, a copy of the notice must '
    'also be simultaneously sent in the same manner to the address for the applicable Genesys entity as set forth at '
    'https://www.genesys.com/company/legal-docs/governing-law-jurisdiction-and-notices. Notices shall be effective when received.'
           }]



inputs = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
)

tokens = model.generate(
    inputs.to(model.device),
    max_new_tokens=1024,
    temperature=0.01,
    do_sample=True
)

print(tokenizer.decode(tokens[0], skip_special_tokens=False))