from transformers import pipeline
import openai
import streamlit as st


def using_BART(
    no_tokens: int, max_tokens_model: int, str_trans_sp: dict, sum_model: str
):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    tokens_used_so_far = 0
    transcription_chunk = ""
    if no_tokens < max_tokens_model:
        summary = summarizer(
            f'""{str_trans_sp}""', max_length=50, min_length=10, do_sample=False
        )
        # print(f"When using {sum_model} summary is:\n {summary}")
    else:
        # print(f"There are more tokens than supported by the model.")
        st.write(
            f"There is more context in the transcript than this model can take in. Will slide the model over the "
            f"allowed context length."
        )
        summaries = []

        total_progress = 0.0
        my_bar = st.progress(total_progress)
        no_sentences = len(str_trans_sp)
        delta_progress = 1.0 / no_sentences

        for idx, sentence in enumerate(str_trans_sp):
            if tokens_used_so_far + len(sentence.split()) <= max_tokens_model:
                transcription_chunk += sentence
                tokens_used_so_far += len(sentence.split())
                to_generate_summary = True
                # print(f"Got here with {tokens_used_so_far}")
            else:
                summary = summarizer(
                    f'""{transcription_chunk}""',
                    max_length=30,
                    min_length=10,
                    do_sample=False,
                )
                summaries.append(summary[0]["summary_text"])
                # print(f"Current summary is {summary[0]}")
                transcription_chunk = sentence
                tokens_used_so_far = len(sentence.split())
            my_bar.progress(min(1.0, total_progress + delta_progress))
            total_progress += delta_progress

        if transcription_chunk != "":
            summary = summarizer(
                f'""{transcription_chunk}""',
                max_length=30,
                min_length=10,
                do_sample=False,
            )
            summaries.append(summary[0]["summary_text"])
        # print(f"List of summaries so far: {summaries}\n")
        concat_summaries = " ".join(summaries)
        st.write("##### Concatenated summary is:")
        st.write(f"#### {concat_summaries}")

        # Now summarising the concatenated summaries using the same model
        summary = summarizer(
            f'""{concat_summaries}""',
            max_length=150,
            min_length=50,
            do_sample=False,
        )
        # print(f"When using {sum_model} final summary is:\n {summary}")
        st.write("##### Final summary is:")
        st.write(f"#### {summary[0]['summary_text']}")
    with open(f"data/bart_summaries/experiment.txt", "w") as f:
        f.write(summary[0]["summary_text"])
    return summary[0]["summary_text"]


def using_GPT3(
    no_tokens: int, max_tokens_model: int, str_trans_sp: str, sum_model, args
):
    openai.api_key = args.openai_key
    tokens_used_so_far = 0
    transcription_chunk = ""
    if no_tokens < max_tokens_model:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"{str_trans_sp}\nTl;dr",
            temperature=0.3,
            max_tokens=120,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        print(
            f"When using {sum_model} summary is:\n\
                {response.choices[0].text}"
        )
    else:
        st.write(
            f"There is more context in the transcript than this model can take in. Will slide the model over the "
            f"allowed context length."
        )
        summaries = []

        total_progress = 0.0
        my_bar = st.progress(total_progress)
        no_sentences = len(str_trans_sp)
        delta_progress = 1.0 / no_sentences

        for sentence in str_trans_sp:
            if tokens_used_so_far + len(sentence.split()) <= max_tokens_model:
                transcription_chunk += sentence
                tokens_used_so_far += len(sentence.split())
                to_generate_summary = True
                # print(f"Got here with {tokens_used_so_far}")
            else:
                response = openai.Completion.create(
                    model="text-davinci-002",
                    prompt=f"{transcription_chunk}\nTl;dr",
                    temperature=0.3,
                    max_tokens=120,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                summaries.append(response.choices[0].text)
                # print(f"Current summary is {response.choices[0].text}")
                transcription_chunk = sentence
                tokens_used_so_far = len(sentence.split())
            my_bar.progress(min(1.0, total_progress + delta_progress))
            total_progress += delta_progress

        if transcription_chunk != "":
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=f"{transcription_chunk}\nTl;dr",
                temperature=0.3,
                max_tokens=120,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            summaries.append(response.choices[0].text)
            # print(f"Current summary is {response.choices[0].text}")
        # print(f"List of summaries so far: {summaries}\n")
        concat_summaries = " ".join(summaries)
        # print(f"no tokens in concat summary is {len(concat_summaries.split())}\n")
        st.write("##### Concatenated summary is:")
        st.write(f"#### {concat_summaries}")

        # Now summarising the concatenated summaries either with Tl;dr or Summarize
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"{concat_summaries}\nTl;dr:",
            temperature=0.6,
            max_tokens=250,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        # print(
        #     f"When using {sum_model} summary is:\n\
        #         {response.choices[0].text}"
        # )
        st.write("##### Concatenated summary is and passed through Tl;dr GPT3:")
        st.write(f"#### {response.choices[0].text}")
        with open(f"data/gpt_summaries/experiment_concat.txt", "w") as f:
            f.write(response.choices[0].text)

        response_sum = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"{concat_summaries}\nSummarize:",
            temperature=0.6,
            max_tokens=250,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        # print(
        #     f"When using {sum_model} summary is:\n\
        #         {response_sum.choices[0].text}"
        # )
        st.write("##### Concatenated summary is and passed through Summarize GPT3:")
        st.write(f"#### {response_sum.choices[0].text}")
        "data/gpt_summaries".mkdir(parents=True, exist_ok=True)
    with open(f"data/gpt_summaries/experiment_short.txt", "w") as f:
        f.write(response_sum.choices[0].text)


def using_Longformer(no_tokens, max_tokens_model, str_trans_sp, sum_model):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    tokens_used_so_far = 0
    transcription_chunk = ""
    if no_tokens < max_tokens_model:
        summary = summarizer(
            f'""{str_trans_sp}""', max_length=50, min_length=10, do_sample=False
        )
        # print(f"When using {sum_model} summary is:\n {summary}")
    else:
        print(f"There are more tokens than supported by the model.")
        summaries = []
        for sentence in str_trans_sp:
            if tokens_used_so_far + len(sentence.split()) <= max_tokens_model:
                transcription_chunk += sentence
                tokens_used_so_far += len(sentence.split())
                to_generate_summary = True
                # print(f"Got here with {tokens_used_so_far}")
            else:
                summary = summarizer(
                    f'""{transcription_chunk}""',
                    max_length=30,
                    min_length=10,
                    do_sample=False,
                )
                summaries.append(summary[0]["summary_text"])
                print(f"Current summary is {summary[0]}")
                transcription_chunk = sentence
                tokens_used_so_far = len(sentence.split())

        if transcription_chunk != "":
            summary = summarizer(
                f'""{transcription_chunk}""',
                max_length=30,
                min_length=10,
                do_sample=False,
            )
            summaries.append(summary[0]["summary_text"])
        print(f"List of summaries so far: {summaries}\n")
        concat_summaries = " ".join(summaries)
        print(f"no tokens in concat summary is {len(concat_summaries.split())}\n")
        summary = summarizer(
            f'""{transcription_chunk}""',
            max_length=150,
            min_length=50,
            do_sample=False,
        )
        print(f"When using {sum_model} final summary is:\n {summary}")
    with open(f"data/bart_summaries/experiment.txt", "w") as f:
        f.write(summary[0]["summary_text"])
    return summary[0]["summary_text"]
