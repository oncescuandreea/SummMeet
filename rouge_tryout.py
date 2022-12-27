from rouge import Rouge


def main():

    with open("data/bart_summaries/ES2008a.abssumm.txt", "r") as f:
        hypothesis = f.read().splitlines()
    with open("data/real_summaries/ES2008a.abssumm.txt", "r") as f:
        reference = f.read().splitlines()
    print(hypothesis[0])
    print("\n")
    print(reference[0])
    rouge = Rouge()
    # scores = rouge.get_scores(
    #     "So I'm sure we'll all have more concrete things to contribute next time",
    #     "The project manager opened the meeting and introduced herself to the team.",
    # )
    # scores = rouge.get_scores(hypothesis[0], reference[0])

    # scores = rouge.get_scores(
    #     "The team is discussing their project aim, which is to create a new remote control, and they are also discussing their experiences \
    #         with using the remote controls and their ideas for what would make a good remote control. The team discussed the idea of \
    #             having two separate remotes, one for the TV and one for the VCR. They also talked about the importance of having large easy-to-press \
    #                 buttons. Alima will be working on the industrial design and the user interface design. The team will be meeting again in 30 minutes.",
    #     reference[0],
    # ) # this is the one where tldr is used for two chunks and they are concatenated. with the speakers info
    # [{'rouge-1': {'r': 0.391304347826087, 'p': 0.4426229508196721, 'f': 0.41538461040355035},
    # 'rouge-2': {'r': 0.14414414414414414, 'p': 0.1951219512195122, 'f': 0.16580310392117922},
    # 'rouge-l': {'r': 0.36231884057971014, 'p': 0.4098360655737705, 'f': 0.3846153796343195}}]

    # scores = rouge.get_scores(
    #     "We are designing a new television remote control. We will be discussing this more in our next meeting. Our indivisual actions and \
    #         then we'll come back together.",
    #     reference[0],
    # )  # this is the one where tldr is used for two chunks and they are concatenated and then summarised using TlDr with the speakers
    # [{'rouge-1': {'r': 0.11594202898550725, 'p': 0.3076923076923077, 'f': 0.16842104865595575},
    # 'rouge-2': {'r': 0.009009009009009009, 'p': 0.038461538461538464, 'f': 0.014598537070702336},
    # 'rouge-l': {'r': 0.08695652173913043, 'p': 0.23076923076923078, 'f': 0.12631578549806108}}]

    # scores = rouge.get_scores(
    #     "The conversation was about the design of a new remote control for a television. The group discussed the idea of having one remote with the main \
    #         functions on it, and another remote with the special functions on it. They also discussed the importance of having large buttons on the remote.",
    #     reference[0],
    # )  # this is the one where tldr is used for two chunks and they are concatenated and then summarised using Summarize with the speakers
    # # [{'rouge-1': {'r': 0.15942028985507245, 'p': 0.34375, 'f': 0.21782177784923049},
    # # 'rouge-2': {'r': 0.04504504504504504, 'p': 0.11363636363636363, 'f': 0.06451612496649349},
    # # 'rouge-l': {'r': 0.15942028985507245, 'p': 0.34375, 'f': 0.21782177784923049}}]

    scores = rouge.get_scores(
        "The team is discussing their project aim and financial goals. They then move on to talking about their experiences with using remote controls \
            and what features they would like to see in a new remote control. The team discussed the possibility of designing a remote control with \
                larger buttons, and decided that it would be a good idea to focus on designing a remote control for a telvevision only. They also decided \
                    that they would each work on different aspects of the project, and that they would meet again to discuss their project.",
        reference[0],
    )  # this is the one where tldr is used for two chunks and they are concatenated without using the speakers
    # [{'rouge-1': {'r': 0.36231884057971014, 'p': 0.43103448275862066, 'f': 0.39370078243908496},
    # 'rouge-2': {'r': 0.13513513513513514, 'p': 0.18292682926829268, 'f': 0.15544040962066114},
    # 'rouge-l': {'r': 0.36231884057971014, 'p': 0.43103448275862066, 'f': 0.39370078243908496}}]
    print(scores)


if __name__ == "__main__":
    main()
