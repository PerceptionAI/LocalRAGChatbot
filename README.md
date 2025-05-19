This is a work in progress for completely local, talking chatbot where you can ask question upon providing .txt files that contain the topics for discussion.  You need an NVidia GPU to run this so it will not work on a Mac. 
You will need to upload the Mistral-7B-Instruct-v0.2-GGUF model from https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF.  You also need to provide around 15 seconds or more of a sample voice called speaker_ref.wav.   
This enables you to customize the sound of the voice you want the bot to have.   Currently, the more words the .txt contain, it seems to increase the latency of the answers from the voice bot.  This is simply a prototype and 
it will require a lot more work to make it useful without annoying the user with delayed responses.
