mod shell;
mod model;

use std::collections::HashMap;
use std::io::{self, Write};
use std::fs;
use std::process::Command as ProcessCommand;
use serde::{Deserialize, Serialize};
use clap::{Command, Arg};
use colored::*;
use std::path::PathBuf;
use shell::Shell;
use crate::model::Model;

#[derive(Serialize, Deserialize)]
struct Config {
    model: Model,
    max_tokens: i32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("llm-term")
        .version("1.0")
        .author("dh1101")
        .about("Generate terminal commands using OpenAI or local Ollama models")
        .arg(Arg::new("prompt")
            .help("The prompt describing the desired command")
            .required(false)
            .index(1))
        .arg(Arg::new("config")
            .short('c')
            .long("config")
            .help("Run configuration setup")
            .action(clap::ArgAction::SetTrue))
        .arg(Arg::new("show-config")
            .long("show-config")
            .help("Display current configuration")
            .action(clap::ArgAction::SetTrue))
        .arg(Arg::new("custom-model")
            .long("custom-model")
            .help("Set custom model name")
            .value_name("MODEL_NAME"))
        .arg(Arg::new("custom-endpoint")
            .long("custom-endpoint")
            .help("Set custom endpoint URL")
            .value_name("ENDPOINT_URL"))
        .arg(Arg::new("custom-system-prompt")
            .long("custom-system-prompt")
            .help("Set custom system prompt")
            .value_name("SYSTEM_PROMPT"))
        .arg(Arg::new("custom-api-key")
            .long("custom-api-key")
            .help("Set custom API key")
            .value_name("API_KEY"))
        .arg(
            Arg::new("disable-cache")
                .long("disable-cache")
                .help("Disable cache and always query the LLM")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let config_path = get_default_config_path().expect("Failed to get default config path");

    // Handle custom model configuration
    let custom_model_name = matches.get_one::<String>("custom-model");
    let custom_endpoint = matches.get_one::<String>("custom-endpoint");
    let custom_system_prompt = matches.get_one::<String>("custom-system-prompt");
    let custom_api_key = matches.get_one::<String>("custom-api-key");

    if custom_model_name.is_some() || custom_endpoint.is_some() || custom_system_prompt.is_some() || custom_api_key.is_some() {
        let mut config = load_or_create_config(&config_path)?;
        
        if let (Some(model_name), Some(endpoint)) = (custom_model_name, custom_endpoint) {
            config.model = Model::Custom {
                model_name: model_name.clone(),
                endpoint: endpoint.clone(),
                system_prompt: custom_system_prompt.cloned(),
                api_key: custom_api_key.cloned(),
            };
            let content = serde_json::to_string_pretty(&config)?;
            fs::write(&config_path, content)?;
            println!("{}", "Custom model configuration saved successfully.".green());
        } else {
            println!("{}", "Error: Both --custom-model and --custom-endpoint are required for custom model configuration.".red());
            return Ok(());
        }
        return Ok(());
    }

    if matches.get_flag("config") {
        let config = create_config()?;
        let content = serde_json::to_string_pretty(&config)?;
        fs::write(&config_path, content)?;
        println!("{}", "Configuration saved successfully.".green());
        return Ok(());
    }

    let config = load_or_create_config(&config_path)?;

    // Handle show-config flag
    if matches.get_flag("show-config") {
        let shell = Shell::detect();
        println!("{}", "Current Configuration:".cyan().bold());
        println!("{}", config.model.display_config(&shell));
        println!("{}", format!("Max Tokens: {}", config.max_tokens).cyan());
        return Ok(());
    }

    let cache_path = get_cache_path()?;
    let mut cache = load_cache(&cache_path)?;

    if let Some(prompt) = matches.get_one::<String>("prompt") {
        let disable_cache = matches.get_flag("disable-cache");

        if !disable_cache {
            if let Some(cached_command) = cache.get(prompt) {
                println!("{}", "This command exists in cache".yellow());
                println!("{}", cached_command.cyan().bold());
                println!("{}", "Do you want to execute this command? (y/n)".yellow());

                let mut user_input = String::new();
                io::stdin().read_line(&mut user_input)?;

                if user_input.trim().to_lowercase() == "y" {
                    execute_command(cached_command)?;
                } else {
                    println!("{}", "Do you want to invalidate the cache? (y/n)".yellow());
                    user_input.clear();
                    io::stdin().read_line(&mut user_input)?;

                    if user_input.trim().to_lowercase() == "y" {
                        // Invalidate cache
                        cache.remove(prompt);
                        save_cache(&cache_path, &cache)?;
                        // Proceed to get command from LLM
                        get_command_from_llm(&config, &mut cache, &cache_path, prompt)?;
                    } else {
                        println!("{}", "Command execution cancelled.".yellow());
                    }
                }
                return Ok(());
            } else {
                // Not in cache, proceed to get command from LLM
                get_command_from_llm(&config, &mut cache, &cache_path, prompt)?;
            }
        } else {
            // Cache is disabled, proceed to get command from LLM
            get_command_from_llm(&config, &mut cache, &cache_path, prompt)?;
        }
    } else {
        println!("{}", "Please provide a prompt or use one of the following options:".yellow());
        println!("{}", "  --config                 Set up configuration".cyan());
        println!("{}", "  --show-config           Display current configuration".cyan());
        println!("{}", "  --custom-model          Set custom model name".cyan());
        println!("{}", "  --custom-endpoint       Set custom endpoint URL".cyan());
        println!("{}", "  --custom-system-prompt  Set custom system prompt".cyan());
        println!("{}", "  --custom-api-key        Set custom API key".cyan());
    }

    Ok(())
}

fn get_default_config_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent().ok_or("Failed to get executable directory")?;
    Ok(exe_dir.join("config.json"))
}

fn load_or_create_config(path: &PathBuf) -> Result<Config, Box<dyn std::error::Error>> {
    if let Ok(content) = fs::read_to_string(path) {
        Ok(serde_json::from_str(&content)?)
    } else {
        let config = create_config()?;
        let content = serde_json::to_string_pretty(&config)?;
        fs::write(path, content)?;
        Ok(config)
    }
}

fn create_config() -> Result<Config, io::Error> {
    let model = loop {
        println!("{}", "Select model:\n 1 for gpt-4o-mini\n 2 for gpt-4o\n 3 for ollama (llama3.1)\n 4 for custom model".cyan());

        io::stdout().flush()?;
        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        match choice.trim() {
            "1" => break Model::OpenAiGpt4oMini,
            "2" => break Model::OpenAiGpt4o,
            "3" => break Model::Ollama("llama3.1".to_string()),
            "4" => {
                print!("{}", "Enter custom model name: ".cyan());
                io::stdout().flush()?;
                let mut model_name = String::new();
                io::stdin().read_line(&mut model_name)?;
                let model_name = model_name.trim().to_string();

                print!("{}", "Enter endpoint URL: ".cyan());
                io::stdout().flush()?;
                let mut endpoint = String::new();
                io::stdin().read_line(&mut endpoint)?;
                let endpoint = endpoint.trim().to_string();

                print!("{}", "Enter custom system prompt (optional, press Enter to use default): ".cyan());
                io::stdout().flush()?;
                let mut system_prompt = String::new();
                io::stdin().read_line(&mut system_prompt)?;
                let system_prompt = if system_prompt.trim().is_empty() {
                    None
                } else {
                    Some(system_prompt.trim().to_string())
                };

                print!("{}", "Enter API key (optional, press Enter to use environment variable): ".cyan());
                io::stdout().flush()?;
                let mut api_key = String::new();
                io::stdin().read_line(&mut api_key)?;
                let api_key = if api_key.trim().is_empty() {
                    None
                } else {
                    Some(api_key.trim().to_string())
                };

                break Model::Custom {
                    model_name,
                    endpoint,
                    system_prompt,
                    api_key,
                };
            },
            _ => println!("{}", "Invalid choice. Please try again.".red()),
        }
    };

    let max_tokens = loop {
        print!("{}", "Enter max tokens (1-4096): ".cyan());
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if let Ok(tokens) = input.trim().parse::<i32>() {
            if tokens > 0 && tokens <= 4096 {
                break tokens;
            }
        }
        println!("{}", "Invalid input. Please enter a number between 1 and 4096.".red());
    };

    Ok(Config {
        model,
        max_tokens,
    })
}

fn get_cache_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent().ok_or("Failed to get executable directory")?;
    Ok(exe_dir.join("cache.json"))
}

fn load_cache(path: &PathBuf) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    if let Ok(content) = fs::read_to_string(path) {
        Ok(serde_json::from_str(&content)?)
    } else {
        Ok(HashMap::new())
    }
}

fn save_cache(path: &PathBuf, cache: &HashMap<String, String>) -> Result<(), Box<dyn std::error::Error>> {
    let content = serde_json::to_string_pretty(&cache)?;
    fs::write(path, content)?;
    Ok(())
}

fn get_command_from_llm(
    config: &Config,
    cache: &mut HashMap<String, String>,
    cache_path: &PathBuf,
    prompt: &String,
) -> Result<(), Box<dyn std::error::Error>> {
    match &config.model.llm_get_command(config, prompt.as_str()) {
        Ok(Some(command)) => {
            println!("{}", &command.cyan().bold());
            println!("{}", "Do you want to execute this command? (y/n)".yellow());

            let mut user_input = String::new();
            io::stdin().read_line(&mut user_input)?;

            if user_input.trim().to_lowercase() == "y" {
                execute_command(&command)?;
            } else {
                println!("{}", "Command execution cancelled.".yellow());
            }

            // Save command to cache
            cache.insert(prompt.clone(), command.clone());
            save_cache(cache_path, cache)?;
        },
        Ok(None) => println!("{}", "No command could be generated.".yellow()),
        Err(e) => eprintln!("{}", format!("Error: {}", e).red()),
    }

    Ok(())
}

fn execute_command(command: &str) -> Result<(), Box<dyn std::error::Error>> {
    let (shell_cmd, shell_arg) = Shell::detect().to_shell_command_and_command_arg();

    match ProcessCommand::new(shell_cmd).arg(shell_arg).arg(&command).output() {
        Ok(output) => {
            println!("{}", "Command output:".green().bold());
            io::stdout().write_all(&output.stdout)?;
            io::stderr().write_all(&output.stderr)?;
        }
        Err(e) => eprintln!("{}", format!("Failed to execute command: {}", e).red()),
    }

    Ok(())
}