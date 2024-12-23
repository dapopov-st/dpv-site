import logging
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from ts.torch_handler.base_handler import BaseHandler

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class TransformersHandler(BaseHandler):
    def initialize(self, ctx):
        logger.info("Initializing the TransformersHandler.")
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model and tokenizer loaded successfully.")

    def preprocess(self, data):
        logger.info("Preprocessing input data.")
        logger.debug(f"Raw data received: {data}")
        logger.info(f"DATA: {data}")
        data_body = data[0]['body']
        logger.info(f"DATA_BODY: {data_body}")

        text = data_body.get("text")
        if isinstance(text, bytes):
            text = text.decode('utf-8')
            logger.debug(f"Decoded text: {text}")
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            logger.debug(f"Tokenized inputs: {inputs}")
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            raise e
        
        return inputs

    def inference(self, inputs):
        logger.info("Performing inference.")
        try:
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidences, predictions = torch.max(probs, dim=1)
                result = {"confidence": confidences.item(), 
                          "prediction": predictions.item()}
                logger.debug(f"Inference result: {result}")
                return result
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise e

    def postprocess(self, inference_output):
        logger.info("Postprocessing inference output.")
        logger.debug(f"Postprocessing result: {inference_output}")
        return [inference_output]