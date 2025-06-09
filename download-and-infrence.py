from diffusers import DiffusionPipeline
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import os
import sys
import torch  # Import torch to check for GPU availability

# Initialize the console
console = Console()

# Display a welcome banner
def display_banner():
    banner = """
 ██████╗ ██████╗ ██╗   ██╗███████╗███████╗███████╗██╗   ██╗██╗  ██╗██╗     
██╔═══██╗██╔══██╗╚██╗ ██╔╝██╔════╝██╔════╝██╔════╝╚██╗ ██╔╝╚██╗██╔╝██║     
██║   ██║██║  ██║ ╚████╔╝ ███████╗███████╗█████╗   ╚████╔╝  ╚███╔╝ ██║     
██║   ██║██║  ██║  ╚██╔╝  ╚════██║╚════██║██╔══╝    ╚██╔╝   ██╔╝██╗ ██║     
╚██████╔╝██████╔╝   ██║   ███████║███████║███████╗   ██║   ██╔╝ ██╗███████╗
 ╚═════╝ ╚═════╝    ╚═╝   ╚══════╝╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
                                                                           
    OdysseyXL Model Inference Script v1.2 - By Aayan Mishra
    """
    console.print(Panel(banner, style="bold cyan"))

def choose_model():
    table = Table(title="Available OdysseyXL Models", style="bold green")
    table.add_column("Option", justify="center")
    table.add_column("Model Name", justify="left")

    models = {
        1: "Spestly/OdysseyXL-Origin",
        2: "Spestly/OdysseyXL-V1",
        3: "Spestly/OdysseyXL-V2",
        4: "Spestly/OdysseyXL-V2.5",
    }

    for option, model in models.items():
        table.add_row(str(option), model)

    console.print(table)

    choice = console.input("[bold yellow]Select a model by entering its option number:[/] ")

    try:
        choice = int(choice)
        if choice in models:
            return models[choice]
        else:
            raise ValueError
    except ValueError:
        console.print("[bold red]Invalid choice. Please try again.[/]")
        return choose_model()

def select_device():
    if not torch.cuda.is_available():
        console.print("[bold red]No GPUs detected. Using CPU.[/]")
        return "cpu"

    gpu_count = torch.cuda.device_count()
    console.print(f"[bold blue]{gpu_count} GPU(s) detected.[/]")

    if gpu_count == 1:
        console.print("[bold green]Using the single available GPU.[/]")
        return "cuda:0"

    # If multiple GPUs are available, prompt the user
    choice = console.input("[bold yellow]Use all available GPUs? (yes/no): [/] ").strip().lower()

    if choice in ["yes", "y"]:
        console.print("[bold green]Using all available GPUs for inference.[/]")
        return "cuda"  # Use all GPUs (data parallelism can be added if needed)

    console.print("[bold green]Using the first GPU only.[/]")
    return "cuda:0"  # Use only the first GPU

def download_and_infer(model_name, device):
    console.print(f"\n[bold blue]Downloading model: [cyan]{model_name}[/cyan]...[/bold blue]")
    try:
        # Load the model pipeline and set to the selected device
        pipe = DiffusionPipeline.from_pretrained(model_name)
        pipe.to(device)  # Move the pipeline to the appropriate device
        pipe.enable_sequential_cpu_offload()
        console.print(f"[bold green]Successfully downloaded {model_name} to {device.upper()}![/bold green]\n")

        # Prompt to check if the user wants to use a LoRA
        use_lora = console.input("[bold yellow]Do you want to use a custom LoRA? (yes/no): [/] ").strip().lower()
        if use_lora in ["yes", "y"]:
            lora_path = console.input("[bold yellow]Enter the path to the SDXL LoRA:[/] ").strip()
            if os.path.exists(lora_path):
                console.print(f"[bold blue]Loading LoRA from: [cyan]{lora_path}[/cyan]...[/bold blue]")
                pipe.load_lora_weights(lora_path)
                console.print("[bold green]LoRA successfully applied![/bold green]")
            else:
                console.print("[bold red]Invalid LoRA path. Proceeding without LoRA.[/bold red]")

        # Prompt for inference
        prompt = console.input("[bold yellow]Enter your text prompt for inference:[/] ")
        console.print("[bold blue]Generating image...[/bold blue]")

        # Perform inference
        image = pipe(prompt).images[0]

        # Save the result
        output_path = os.path.join(os.getcwd(), f"{model_name.split('/')[-1]}_output.png")
        image.save(output_path)
        console.print(f"[bold green]Image generated and saved at: {output_path}[/bold green]")

    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}[/bold red]")

def main():
    display_banner()
    model_name = choose_model()
    device = select_device()
    download_and_infer(model_name, device)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting... Goodbye![/bold red]")
        sys.exit(0)
