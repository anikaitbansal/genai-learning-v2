import uuid
import argparse
from memory_manager import MemoryManager
from chat_service import ChatService
from retriever import FAISSRetriever
from app_database import initialize_database

retriever = FAISSRetriever()


def read_multiline_input():
    print("User (type DONE on a new line to finish or exit to quit):")
    lines = []

    while True:
        line = input()
        if line.strip() == "DONE":
            break
        lines.append(line)

    return "\n".join(lines).strip()


class Chatbot:
    def __init__(self, debug=False): # debug is a boolean parameter that allows us to enable or disable debug mode when creating an instance of the Chatbot class. When debug
        self.memory = MemoryManager("cli-session") # This line creates an instance of the MemoryManager class and assigns
        self.debug = debug 
        self.service = ChatService(self.memory,retriever=retriever, debug=self.debug)



    def run(self):    
        while True:
            user_input = read_multiline_input()
            if self.debug: # If debug mode is enabled, we print the user's input to the console for debugging purposes. This allows us to see exactly what the user has entered before we process it, which can be helpful for troubleshooting and understanding the flow of the program.
                print("[DEBUG] User input:", user_input)

            if user_input.lower() == "exit":
                print("Exiting the program.")
                break

            if user_input.lower() == "reset":
                self.memory.clear()
                print("Bot: Memory cleared.")
                continue
            
            request_id = str(uuid.uuid4()) 
            result = self.service.process_message(user_input, "cli-session", request_id)
            print("Bot:", result["bot_reply"])

            if not self.debug:
                print("Intent:", result["intent"])


#day 5 additions are everything below.

# class Chatbot:
#    def __init__(self):
#        pass
#    def run(self):

def main(debug=False): # This is the main function that serves as the entry point of the program. It takes an optional debug parameter that allows us to enable or disable debug mode when running the chatbot. If debug mode is enabled, it prints a message indicating that debug mode is on. Then, it creates an instance of the Chatbot class, passing the debug parameter to its constructor, and calls the run method to start the chatbot's interaction loop.
    initialize_database()
    if debug:
        print("Debug mode is ON.")
    bot = Chatbot(debug)
    bot.run()

if __name__ == "__main__": # This is a common Python idiom that checks if the script is being run directly (as the main program) rather than imported as a module in another script. If this condition is true, it executes the code block that follows, which in this case is responsible for parsing command-line arguments and calling the main function with the appropriate debug flag based on the user's input.
    parser = argparse.ArgumentParser() # This line creates an ArgumentParser object from the argparse module, which is used to handle command-line arguments. The parser allows us to define what arguments our program accepts and how they should be processed. In this case, we will use it to add a debug flag that can be set when running the script from the command line.
    parser.add_argument("--debug", action="store_true", help="Enable debug mode") # This line adds a command-line argument called --debug to the parser. The action="store_true" parameter means that if the user includes the --debug flag when running the script, the debug variable will be set to True. If the flag is not included, debug will be False by default. The help parameter provides a description of what the --debug flag does, which will be displayed when the user runs the script with the --help option. This allows users to easily enable debug mode when running the chatbot for troubleshooting or development purposes.
    args = parser.parse_args() # This line parses the command-line arguments that were defined using the parser. It processes the arguments passed to the script when it is run and stores the results in the args variable. In this case, if the user included the --debug flag, args.debug will be set to True; otherwise, it will be False. This allows us to control whether debug mode is enabled when we call the main function.

    main(debug=args.debug) # This line calls the main function, passing the value of args.debug as the debug parameter. This means that if the user included the --debug flag when running the script, debug mode will be enabled in the main function, which will then be passed to the Chatbot instance to control whether debug information is printed during the chatbot's operation.
    