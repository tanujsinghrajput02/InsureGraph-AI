from transformers import AutoTokenizer, AutoModelForCausalLM


class LazyModelLoader:
    """
    Loads the local language model only when it is first requested.
    Reuses the same model and tokenizer for subsequent requests.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def get_model_and_tokenizer(self):
        if not self.loaded:
            print(
                "\nLoading local model for the first LLM request:",
                self.model_name
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            )

            model.eval()

            self.tokenizer = tokenizer
            self.model = model
            self.loaded = True

            print("Local model loaded successfully.")

        return self.model, self.tokenizer