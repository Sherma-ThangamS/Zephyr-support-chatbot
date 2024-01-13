# Zephyr Support Chatbot  

## Overview

 In this project, I fine tuned zephyr-7B-alpha-GPTQ which is the quantized model using Q-Lora technique with the help of SFTTrainer from trl.


## Objective

The primary goal of this project is to train a support chatbot using the Zephyr 7B model, customized for a specific customer support use case. The chatbot is designed to respond to user queries in a professional manner, surpassing the performance of the Llama2-70b-chat model.

## Pre-trained Model on Hugging Face Model Hub

 I have released a pre-trained version of the Zephyr Support Chatbot on the Hugging Face Model Hub. You can find it [here](https://huggingface.co/sst1/zephyr-support-chatbot). Feel free to use this pre-trained model for quick integration into your projects.

To use the pre-trained model, you can load it using the Hugging Face Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "sst1/zephyr-support-chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

## Key Components

1. **Dataset:**
   - Training data sourced from "bitext/Bitext-customer-support-llm-chatbot-training-dataset."
   - Data processing involves creating a prompt for a support chatbot and combining instruction and target fields for each example.

2. **Model Preparation:**
   - Zephyr 7B model loaded and configured for fine-tuning.
   - Quantization applied to the model using a specified number of bits.
   - LoRA (Lora Adapter) modules attached to the model for enhanced performance.
   - Configuration parameters, such as dropout rate and task type, set for LoRA.

3. **Training:**
   - Dataset loaded and processed for training.
   - Model trained using the SFTTrainer from the trl library.
   - Training arguments, such as batch size and learning rate, configured.
   - Training loop runs for a specified number of epochs or steps.

4. **Evaluation:**
   - Trained model can be evaluated on additional datasets or specific test cases.

5. **Inference:**
   - After training, the model can generate responses to user queries.
   - Inference involves tokenizing input queries and using the trained model to generate appropriate responses.

6. **Output:**
   - Project outputs include a fine-tuned Zephyr 7B model and associated artifacts.
   - Model performance metrics, such as loss and accuracy, can be analyzed.

7. **Integration with Hugging Face Hub:**
   - Project leverages the Hugging Face Hub for model sharing and version control.
   - Trained model can be pushed to the Hugging Face Hub for community access.

## Usage

- The project is designed to be used as a standalone script or integrated into other applications.
- Users can customize configuration parameters and train the model on their specific datasets.

## Dependencies

- Python libraries: transformers, trl, peft, accelerate, bitsandbytes, auto-gptq, optimum.


## Training

- Run the training script: `FineTunning.ipynb`
- Monitor training progress and adjust parameters as needed.

## Evaluation

- Evaluate the trained model on test datasets or specific test cases.

## Inference

- Use the trained model for generating responses to user queries.

## Output

- Access the fine-tuned Zephyr 7B model and associated artifacts in the specified output directory.

## Integration with Hugging Face Hub

- Push the trained model to the Hugging Face Hub for community access.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome. For major changes, please open an issue first to discuss the proposed changes.

## Acknowledgments

- Acknowledgments or credits to any external sources or libraries used in the project.
