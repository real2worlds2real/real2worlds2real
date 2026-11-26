"""Load SAM model/processor pairs; caller owns prompts and acceptance thresholds."""

def load_sam3(device="cuda", **model_options):
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained("facebook/sam3", **model_options).to(device).eval()
    return model, Sam3Processor.from_pretrained("facebook/sam3")
