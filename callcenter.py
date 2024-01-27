#C:\Users\Deshan Sumanathilaka\OneDrive - Swansea University\LLM\Week-06\CODES\calls\mp3\Health-Insurance-1.mp3

#C:\Users\Deshan Sumanathilaka\OneDrive - Swansea University\LLM\Week-06\CODES\calls\mp3\Travel-Reservation.mp3

from openai import OpenAI
import os, yaml
import streamlit as st
from htmlTemplates import css,user_template

with open('credentials.yaml') as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)

os.environ['OPENAI_API_KEY'] = credentials['OPENAI_API_KEY']

client = OpenAI()

def complete_model(USER_MESSAGE):
    response = client.chat.completions.create(
                                            model = 'gpt-3.5-turbo',
                                            messages = [
                                                        {"role": "system", "content" : "You are a helpful assitant to solve WSD"},
                                                        {"role": "user", "content": USER_MESSAGE}              
                                                        ],
                                            temperature=0,
                                            max_tokens=1500
                                            )
    return str(response.choices[0].message.content)

def audioToText(audio_file):
    audio_file=open(audio_file,"rb")
    transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    response_format="text" 
                )
    return transcript

def result_generator(transcript):   
    prompt=f'Analyse the content below. Return the overall sentiment as positive, negtive or neutral. give me a one word answer  "{transcript}."' 
    result=complete_model(prompt)

    prompt=f'Analyse the content below and summarize it. {transcript}'
    summary=complete_model(prompt)

    prompt=f'Analyse the conversation below. How good was the service given by the call center agent. Arrange the answer in a manner of good and improvements. {transcript}. ' 
    quality=complete_model(prompt)

    return result,summary,quality
    
def Pipeline(url):
    
    transcript=audioToText(url)
    result,summary,quality=result_generator(transcript)
    return result,summary,quality

def main():
    st.set_page_config(page_title="Call center analyser",
                       page_icon=":iphone:")
    st.write(css, unsafe_allow_html=True)

    st.header("Call Center Analyser :iphone:")
    url = st.text_input("Share the URL of the Audio file")
    
    if st.button("Analyse"):
        with st.spinner("Searching for the best answer"):
            result,summary,quality=Pipeline(url)
            st.subheader("Overall Sentiment :")
            result=result.lower
            if result == "positive":
                st.write("Positive :smiley:")
            elif result == "negative":
                st.write("Negative :unamused:")
            else:
                st.write("Neutral :neutral_face:")

            st.subheader("Summary of the Discussion :")
            st.write(summary)

            st.subheader("Quality of the Discussion")
            st.write(quality)

          
                
if __name__ == '__main__':
    main()


    


        