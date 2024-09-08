"""Trains a PyTorch image classification model using device-agnostic code"""

"""Trains a PyTorch image classification model using device-agnostic code"""

import os
import torch
from torchvision import transforms
from timeit import default_timer as timer 
import data_setup, engine, model_builder, utils

def main():
    # Setup Hyperparameters
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001
    
    # Setup directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"
    
    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create transforms
    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])
    
    # Create DataLoaders and get class_names
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                  test_dir=test_dir,
                                                                                  transform=data_transform,
                                                                                  batch_size=BATCH_SIZE)
    # Create model
    model = model_builder.TinyVGG(input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names))

    # If multiple GPUs are available, wrap model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model) # Distribute the model on available GPUs

    model = model.to(device)
    
    # Setup loss and optimizers
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Start the timer
    start_time = timer()
    
    # Start training with help from engine.py
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    # End the timer and print out how long it took to train
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
    
    # Save the model to file
    utils.save_model(model=model,
                    target_dir="models",
                    model_name="going_modular_script_mode_tinyvgg_model.pth")
  
if __name__ == '__main__':
    main()
