# Chatbot 🤖
***A simple **chatbot** implemented in Python.  
It uses **regular expressions** for keyword matching, cleans user input to ignore punctuation, and provides **randomized responses** for a more natural conversation.***



## Features
- Responds to greetings, small talk, and polite expressions.
- Tells jokes from a small predefined list.
- Uses **randomized responses** (no repeated answers).
- Cleans input (so `hoW ArE you!?` = `how are you`).
- Provides default fallback if input is not recognized.
- Lightweight (no external dependencies).



## Conversation Flow
1. User types a message.  
2. The bot cleans the input (removes punctuation, lowercases).  
3. Regex checks for keywords.  
4. If found → bot replies with a random response from the matching list.  
5. If not found → bot replies with a default “I don’t understand” response.  
6. Conversation continues until the user types **exit**.  


## Example Conversation

```
Welcome to the Chatbot! Type 'exit' to end the conversation.

You: hello
Chatbot: Hey! Glad you're here

You: how old are you?
Chatbot: I don't have an age, I just run on code.

You: how are you?
Chatbot: I'm always good when I'm chatting with you.

You: who created you?
Chatbot: Kareem Samy made me to help you out.

You: favorite food?
Chatbot: I don't eat, but I've heard pizza is a favorite.

You: joke
Chatbot: Why don't programmers like nature? Too many bugs!

You: thanks
Chatbot: Glad I could help!

You: bye
Chatbot: See you later! Take care.

You: exit
Chatbot: Goodbye! Have a great day!
```



## Customization
- Add more keywords and responses inside `responses` dictionary.  
- Add synonyms for better matching.  
- Extend with external APIs to make it smarter.  



## Outcome
1. A running chatbot in your terminal.  
2. Randomized replies for a more natural conversation.  
3. A fallback system for unrecognized inputs.  


