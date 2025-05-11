# --- main_s2s_rag_app.py ---

import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from llama_cpp import Llama
from TTS.api import TTS as CoquiTTS # Renamed to avoid conflict
from sentence_transformers import SentenceTransformer
import faiss # For RAG

# --- Configuration ---
# ASR (Faster Whisper)
WHISPER_MODEL_SIZE = "base.en"  # Options: "tiny.en", "base.en", "small.en", "medium.en", "large-v2", "large-v3"
WHISPER_DEVICE = "cuda" # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE = "float16" # "float16" for GPU, "int8" for CPU (or "float32")

# LLM (Llama.cpp)
# Download GGUF model (e.g., from TheBloke on HuggingFace)
# Example: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# OR Dolphin: https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF
LLM_MODEL_PATH = "./models/dolphin-2.2.1-mistral-7b.Q4_K_M.gguf" # <--- SET THIS PATH
N_GPU_LAYERS = 35  # Number of layers to offload to GPU. Set to 0 for CPU only. Adjust based on VRAM.
N_CTX = 2048     # Context window size for LLM

# RAG
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
KNOWLEDGE_BASE_DIR = "./knowledge_base" # Directory with .txt files for RAG
FAISS_INDEX_PATH = "./faiss_index.idx"
MAX_RAG_RESULTS = 3

# TTS (Coqui XTTSv2)
# XTTSv2 will download models on first run if not found.
# You'll need a speaker reference WAV file for voice cloning.
# Get a clean, short (<15s) audio clip of the desired voice.
TTS_SPEAKER_WAV = "./speaker_ref.wav" # <--- CREATE OR PROVIDE THIS
TTS_LANGUAGE = "en"
TTS_DEVICE = "cuda" # "cuda" or "cpu"

# Audio Recording
SAMPLE_RATE = 16000 # Whisper prefers 16kHz
CHANNELS = 1
RECORD_DURATION = 7 # Max recording duration in seconds (can be adjusted)

# --- Global Variables ---
asr_model = None
llm = None
tts_model = None
embedding_model = None
faiss_index = None
doc_chunks = [] # Store original text chunks for RAG

# --- Initialization Functions ---
def init_asr():
    global asr_model
    print("Loading ASR model...")
    try:
        asr_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        print("ASR model loaded.")
    except Exception as e:
        print(f"Error loading ASR model: {e}")
        print("Ensure you have CUDA installed if using GPU, or set WHISPER_DEVICE='cpu'.")
        exit()

def init_llm():
    global llm
    print("Loading LLM...")
    if not os.path.exists(LLM_MODEL_PATH):
        print(f"LLM model not found at {LLM_MODEL_PATH}. Please download it.")
        print("Example: TheBloke/dolphin-2.2.1-mistral-7B-GGUF from HuggingFace.")
        exit()
    try:
        llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            verbose=False # Set to True for more llama.cpp output
        )
        print("LLM loaded.")
    except Exception as e:
        print(f"Error loading LLM: {e}")
        print("Ensure llama-cpp-python is installed correctly and the model path is valid.")
        exit()

def init_tts():
    global tts_model
    print("Loading TTS model...")
    if not os.path.exists(TTS_SPEAKER_WAV):
        print(f"TTS Speaker reference WAV not found at {TTS_SPEAKER_WAV}")
        print("Please provide a clean, short (<15s) audio clip of the desired voice.")
        # Create a dummy one for now if you want to proceed without a good voice
        # For actual use, replace this with a real recording.
        print("Creating a dummy speaker_ref.wav (silent). Replace for actual voice cloning.")
        dummy_audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32) # 2 seconds of silence
        sf.write(TTS_SPEAKER_WAV, dummy_audio, SAMPLE_RATE)
        # exit() # You might want to exit if a real speaker wav is crucial

    try:
        # Check available models: CoquiTTS.list_models()
        # XTTSv2 is "tts_models/multilingual/multi-dataset/xtts_v2"
        tts_model = CoquiTTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(TTS_DEVICE)
        print("TTS model loaded.")
    except Exception as e:
        print(f"Error loading TTS model: {e}")
        print("Ensure Coqui TTS is installed and network access is available for model download on first run.")
        exit()


def init_rag():
    global embedding_model, faiss_index, doc_chunks
    print("Initializing RAG...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(KNOWLEDGE_BASE_DIR + "/doc_chunks.npy"):
        print("Loading existing FAISS index and document chunks...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        doc_chunks = np.load(KNOWLEDGE_BASE_DIR + "/doc_chunks.npy", allow_pickle=True).tolist()
        if not doc_chunks: # Handle case where loaded array might be empty from a previous error
             print("Warning: Loaded doc_chunks is empty. Re-indexing.")
             build_rag_index() # Attempt to rebuild
        else:
            print(f"Loaded {len(doc_chunks)} document chunks and FAISS index.")

    else:
        print("Building RAG index from knowledge base...")
        build_rag_index()

def build_rag_index():
    global faiss_index, doc_chunks
    doc_chunks = []
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        # Create a dummy knowledge file if it's empty
        with open(os.path.join(KNOWLEDGE_BASE_DIR, "example_knowledge.txt"), "w") as f:
            f.write("The sky is blue during the day.\n")
            f.write("Mistral 7B is a large language model.\n")
            f.write("Python is a versatile programming language.\n")
        print(f"Created dummy knowledge file in {KNOWLEDGE_BASE_DIR}. Add your .txt files there.")


    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(KNOWLEDGE_BASE_DIR, filename), 'r', encoding='utf-8') as f:
                # Simple chunking: one chunk per line. You can implement more sophisticated chunking.
                for line in f:
                    line = line.strip()
                    if line:
                        doc_chunks.append(line)

    if not doc_chunks:
        print("No documents found in knowledge base. RAG will not provide context.")
        # Create a dummy entry to prevent FAISS from failing on empty input
        doc_chunks.append("This is a dummy document for an empty knowledge base.")

    print(f"Found {len(doc_chunks)} chunks for RAG.")
    if doc_chunks:
        embeddings = embedding_model.encode(doc_chunks, convert_to_tensor=False)
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension) # Using L2 distance
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        np.save(KNOWLEDGE_BASE_DIR + "/doc_chunks.npy", np.array(doc_chunks, dtype=object))
        print("FAISS index built and saved.")
    else:
        # Handle case where doc_chunks is still empty (e.g., only dummy was added and then logic failed)
        faiss_index = None # Ensure faiss_index is None if no real data
        print("No usable document chunks to build FAISS index.")


# --- Core Functions ---
def record_audio(filename="temp_input.wav", duration=RECORD_DURATION, fs=SAMPLE_RATE):
    print(f"Recording for {duration} seconds... Speak now!")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=CHANNELS, dtype='float32')
        sd.wait()  # Wait until recording is finished
        sf.write(filename, recording, fs)
        print(f"Recording saved to {filename}")
        return filename, recording
    except Exception as e:
        print(f"Error during recording: {e}")
        return None, None

def transcribe_audio(audio_path):
    if not asr_model or not audio_path:
        return "Error: ASR model not loaded or no audio path."
    print("Transcribing audio...")
    try:
        segments, info = asr_model.transcribe(audio_path, beam_size=5, language="en") # Added language for clarity
        transcription = "".join([segment.text for segment in segments])
        print(f"Transcription: {transcription}")
        return transcription.strip()
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"Transcription error: {e}"


def retrieve_from_rag(query_text, top_k=MAX_RAG_RESULTS):
    if not faiss_index or not embedding_model or not doc_chunks:
        print("RAG not initialized or no documents. Skipping context retrieval.")
        return ""
    if not query_text:
        return ""
    print(f"Retrieving context for: {query_text}")
    query_embedding = embedding_model.encode([query_text])
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    context = ""
    if indices.size > 0:
        for i in indices[0]:
            if 0 <= i < len(doc_chunks): # Check index bounds
                context += doc_chunks[i] + "\n"
    print(f"Retrieved context:\n{context.strip()}")
    return context.strip()

def generate_llm_response(user_query, rag_context=""):
    if not llm:
        return "Error: LLM not loaded."
    print("Generating LLM response...")

    # Constructing a prompt for Dolphin/Mistral-Instruct style
    # You might need to adjust this based on the specific model's fine-tuning.
    if rag_context:
        prompt = f"""<|im_start|>system
You are a helpful AI assistant. Use the provided context to answer the user's question. If the context doesn't contain the answer, state that you don't have information about it from the provided context. Be concise.
Context:
{rag_context}<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
"""
    else:
        prompt = f"""<|im_start|>system
You are a helpful AI assistant. Answer the user's question concisely.<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
"""
    # print(f"LLM PROMPT:\n{prompt}") # For debugging

    try:
        output = llm(
            prompt,
            max_tokens=250, # Adjust as needed
            stop=["<|im_end|>", "user:", "assistant:"], # Stop sequences for Dolphin
            echo=False # Don't echo the prompt in the output
        )
        response_text = output['choices'][0]['text'].strip()
        print(f"LLM Raw Response: {response_text}")
        return response_text
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return f"LLM generation error: {e}"

def text_to_speech(text, output_filename="temp_output.wav"):
    if not tts_model:
        return "Error: TTS model not loaded."
    if not text:
        return "Error: No text to synthesize."
    print("Synthesizing speech...")
    try:
        # XTTSv2 requires speaker_wav and language
        tts_model.tts_to_file(
            text=text,
            speaker_wav=TTS_SPEAKER_WAV,
            language=TTS_LANGUAGE,
            file_path=output_filename
        )
        print(f"Speech synthesized to {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error during TTS synthesis: {e}")
        return f"TTS synthesis error: {e}"

def play_audio(filename):
    if not filename or not os.path.exists(filename):
        print(f"Audio file not found: {filename}")
        return
    try:
        data, fs = sf.read(filename, dtype='float32')
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")

# --- Main Application Loop ---
def main():
    # Create models directory if it doesn't exist
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    print("Initializing application components...")
    init_asr()
    init_llm()
    init_rag() # Initialize RAG after LLM (or before, order doesn't strictly matter here)
    init_tts()
    
    print("\nApplication Ready!")
    print("--------------------")

    try:
        while True:
            input("Press Enter to start recording, or Ctrl+C to exit...")
            
            # 1. Record Audio
            start_time = time.time()
            audio_file_path, _ = record_audio()
            if not audio_file_path:
                continue
            record_time = time.time() - start_time
            print(f"Recording took {record_time:.2f}s")

            # 2. Speech to Text
            stt_start_time = time.time()
            user_text = transcribe_audio(audio_file_path)
            if not user_text or "error" in user_text.lower():
                print(f"Could not transcribe: {user_text}")
                play_audio(text_to_speech("Sorry, I could not understand what you said."))
                continue
            stt_time = time.time() - stt_start_time
            print(f"STT took {stt_time:.2f}s")

            # 3. RAG Context Retrieval
            rag_start_time = time.time()
            context = retrieve_from_rag(user_text)
            rag_time = time.time() - rag_start_time
            print(f"RAG retrieval took {rag_time:.2f}s")

            # 4. LLM Generation
            llm_start_time = time.time()
            llm_response = generate_llm_response(user_text, context)
            if "error" in llm_response.lower():
                play_audio(text_to_speech("Sorry, I encountered an error generating a response."))
                continue
            llm_time = time.time() - llm_start_time
            print(f"LLM generation took {llm_time:.2f}s")

            # 5. Text to Speech
            tts_start_time = time.time()
            output_audio_file = text_to_speech(llm_response)
            if "error" in output_audio_file.lower() or not os.path.exists(output_audio_file):
                 print(f"Could not synthesize speech: {output_audio_file}")
                 # Fallback: print the text if TTS fails
                 print(f"LLM Response (text only): {llm_response}")
                 continue
            tts_time = time.time() - tts_start_time
            print(f"TTS synthesis took {tts_time:.2f}s")

            # 6. Play Response
            play_audio(output_audio_file)
            
            total_time = time.time() - start_time
            print(f"Total turn time: {total_time:.2f}s\n")

    except KeyboardInterrupt:
        print("\nExiting application.")
    finally:
        # Clean up temporary files if any persist
        if os.path.exists("temp_input.wav"): os.remove("temp_input.wav")
        if os.path.exists("temp_output.wav"): os.remove("temp_output.wav")
        print("Cleanup complete.")

if __name__ == "__main__":
    main()