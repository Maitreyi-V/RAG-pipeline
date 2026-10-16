#!/usr/bin/env python3
"""
Task A4 / Experiment E7 Benchmark Generator:
Builds the 150 held-out test queries across Bhagavad Gita Chapter 2 (BG 2.1..2.72):
  1. 50 Conceptual Questions (Q_CON_01 .. Q_CON_50)
  2. 40 Pasted Sanskrit Shlokas with Noisy Spelling (Q_SHL_01 .. Q_SHL_40)
  3. 35 Pasted Explanation Excerpts (Q_EXP_01 .. Q_EXP_35)
  4. 25 Unanswerable / Out-of-Scope Queries for Abstention (Q_OOS_01 .. Q_OOS_25)

Outputs:
  - data/e7_questions_150.csv
  - data/e7_questions_150.xlsx (if openpyxl is installed)
"""

import csv
import os
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(HERE, "data", "e7_questions_150.csv")
OUT_XLSX = os.path.join(HERE, "data", "e7_questions_150.xlsx")

# Each tuple:
# (question_id, input_mode, question_type, expected_verse, theme_id,
#  question_en_no_ref, expected_answer_en)
QUERIES_150 = [
    # =========================================================================
    # PART 1: 50 CONCEPTUAL QUESTIONS (Q_CON_01 .. Q_CON_50)
    # =========================================================================
    (
        "Q_CON_01", "question", "Interpretive", "BG 2.1, BG 2.2", "T01",
        "How do the lectures contrast compassionate empathy with deluded attachment when analyzing Arjuna's tears on the battlefield?",
        "True compassion does not obstruct dharma, whereas Arjuna's tears stem from moha (deluded bodily attachment) disguised as pity, which Krishna calls kashmala."
    ),
    (
        "Q_CON_02", "question", "Factual", "BG 2.3, BG 2.4", "T01",
        "How does Arjuna respond to Krishna's command to cast off petty weakness of heart, and why does he single out Bhishma and Drona?",
        "Instead of rising to fight, Arjuna objects that he cannot shoot arrows at revered elders like Bhishma and Drona who are worthy of worship (pujarha)."
    ),
    (
        "Q_CON_03", "question", "Interpretive", "BG 2.5, BG 2.6", "T01",
        "Why does Arjuna argue that living on alms is preferable to victory, and what doubt does he express about the outcome of the war?",
        "He says eating begged food is better than enjoying blood-stained worldly pleasures after killing gurus, and adds that they do not even know whether conquering or being conquered is better."
    ),
    (
        "Q_CON_04", "question", "Applied", "BG 2.7, BG 2.8", "T01",
        "When someone realizes that neither career success nor wealth can cure their inner anguish and moral confusion, what step does Arjuna model?",
        "Arjuna admits his nature is overwhelmed by karpanya-dosha and dharmic confusion, surrenders as a disciple (shishyas te'ham), and asks the teacher to instruct him definitively."
    ),
    (
        "Q_CON_05", "question", "Interpretive", "BG 2.9, BG 2.10", "T01",
        "Why is the setting between the two armies significant when Arjuna falls silent and Krishna smiles before beginning his teaching?",
        "Arjuna's silence after declaring 'I will not fight' marks the end of his own reasoning; Krishna's gentle smile (prahasann iva) in the midst of the two armies highlights the irony of Arjuna's grief before imparting wisdom."
    ),
    (
        "Q_CON_06", "question", "Factual", "BG 2.15", "T02",
        "Who does Krishna say becomes fit for immortality (amritatva) in the face of changing sensory experiences?",
        "The wise person (dhira) who remains equal in pain and pleasure (sama-duhkha-sukham) and is not disturbed by sensory contacts becomes fit for immortality."
    ),
    (
        "Q_CON_07", "question", "Interpretive", "BG 2.16", "T02",
        "How do the Advaita and Dvaita commentators differ in interpreting the terms 'sat' and 'asat' in the verse about the real and the unreal?",
        "Advaita interprets sat as eternal Brahman/Atman and asat as the transient world/body that has no enduring existence, whereas Dvaita interprets sat/asat in terms of eternally existent vs non-existent principles or righteous (sadhu) vs unrighteous (asadhu) action."
    ),
    (
        "Q_CON_08", "question", "Factual", "BG 2.17", "T02",
        "How does Krishna describe the principle that pervades this entire universe, and can anyone destroy it?",
        "He describes that which pervades the entire universe as imperishable (avinashi) and immutable (avyaya), stating that no one is capable of destroying it."
    ),
    (
        "Q_CON_09", "question", "Interpretive", "BG 2.18", "T02",
        "What contrast does Krishna draw between the physical bodies and the embodied owner of the body to urge Arjuna to fight?",
        "The physical bodies (deha) are declared to have an end (antavantah), whereas the embodied soul (sharirin) is eternal, indestructible, and immeasurable; therefore Arjuna should fight."
    ),
    (
        "Q_CON_10", "question", "Factual", "BG 2.19", "T02",
        "What error is shared both by the person who thinks the soul is a slayer and by the one who thinks the soul is slain?",
        "Both of them fail to know the truth (ubhau tau na vijanito), because the Atman neither slays nor is ever slain."
    ),
    (
        "Q_CON_11", "question", "Factual", "BG 2.20", "T02",
        "What are the six changes of finite existence (shad-bhava-vikara) that Krishna negates for the soul when describing it as unborn and ancient?",
        "The soul is never born, never dies, does not come into being after not existing, does not grow or decay, and is not killed when the body is killed."
    ),
    (
        "Q_CON_12", "question", "Interpretive", "BG 2.21", "T02",
        "Why does Krishna ask 'whom does such a person cause to kill, and whom does he kill?' regarding a knower of the immutable Self?",
        "A person who knows the Self as indestructible, eternal, unborn, and immutable realizes that the true Self cannot be the object of killing or causing to kill."
    ),
    (
        "Q_CON_13", "question", "Factual", "BG 2.22", "T02",
        "What everyday human habit does Krishna compare to the embodied soul leaving old bodies and entering new ones?",
        "He compares it to a person casting off worn-out garments (vasamsi jirnani) and putting on new clothes."
    ),
    (
        "Q_CON_14", "question", "Factual", "BG 2.23, BG 2.24", "T02",
        "Which four physical elements or forces cannot harm the soul, and what four Sanskrit adjectives summarize this invulnerability?",
        "Weapons cannot cleave it, fire cannot burn it, water cannot wet it, and wind cannot dry it; thus it is acchedya, adahya, akledya, and ashoshya."
    ),
    (
        "Q_CON_15", "question", "Factual", "BG 2.24, BG 2.25", "T02",
        "What does it mean that the soul is sarvagatah, sthanuh, avyakta, and acintya?",
        "It means the soul is all-pervading, stable, unmanifest to the physical senses, inconceivable to the ordinary mind, and unchangeable (avikarya)."
    ),
    (
        "Q_CON_16", "question", "Interpretive", "BG 2.26, BG 2.27", "T03",
        "How does Krishna argue that Arjuna should not grieve even under the hypothetical assumption that the soul is constantly born and constantly dies?",
        "Even if one assumes constant birth and death, death is certain for whatever is born and rebirth is certain for what dies; one should not grieve over an unavoidable (apariharya) law of nature."
    ),
    (
        "Q_CON_17", "question", "Factual", "BG 2.28", "T03",
        "According to Krishna, what are the three states of all beings before birth, during life, and after death?",
        "Beings are unmanifest before birth (avyaktadini), manifest only in the middle state of life (vyaktamadhyani), and unmanifest again upon death (avyaktanidhanani)."
    ),
    (
        "Q_CON_18", "question", "Interpretive", "BG 2.29", "T03",
        "Why does Krishna say that people see, speak of, and hear of the soul as a wonder (ashcharyavat), yet hardly anyone truly knows it?",
        "Because the subtle nature of the Atman transcends ordinary sensory experience, making true realization rare even among those who hear scriptural descriptions."
    ),
    (
        "Q_CON_19", "question", "Cross-verse", "BG 2.11, BG 2.30", "T03",
        "How does Krishna close the first major philosophical section on the indestructibility of the dehin before shifting to kshatriya duty?",
        "Having begun in 2.11 by stating the wise do not grieve, he concludes in 2.30 that because the embodied dweller in every body is eternally invulnerable (nityam avadhyah), Arjuna should not mourn for any creature."
    ),
    (
        "Q_CON_20", "question", "Factual", "BG 2.31, BG 2.32", "T04",
        "Why does Krishna call a righteous war an 'open door to heaven' (svarga-dvaram apavritam) for a Kshatriya?",
        "For a Kshatriya, nothing is more auspicious than a righteous battle fought for dharma; such a duty arrives unsought and grants heaven to fortunate warriors."
    ),
    (
        "Q_CON_21", "question", "Factual", "BG 2.33, BG 2.34", "T04",
        "What moral and social consequences does Krishna warn Arjuna he will suffer if he refuses to fight this dharmya-sangrama?",
        "He will abandon his svadharma and honor, incur sin (papam), and face everlasting dishonor (akirti), which for an esteemed person is worse than death."
    ),
    (
        "Q_CON_22", "question", "Factual", "BG 2.35, BG 2.36", "T04",
        "How will the great chariot-warriors (maharathas) and enemies interpret Arjuna's withdrawal from battle?",
        "They will think he fled out of fear, losing respect for him, and his enemies will utter many unspeakable slanders (avacya-vadan) mocking his prowess."
    ),
    (
        "Q_CON_23", "question", "Interpretive", "BG 2.37", "T04",
        "How is the win-win argument for fighting ('slain you gain heaven, victorious you enjoy the earth') interpreted differently in Dvaita commentary on 'krita-nishcayah'?",
        "While generally read as 'stand up resolved for battle', Dvaita commentary also highlights that the ultimate resolve belongs to Lord Narayana, so Arjuna should act aligned with God's certainty."
    ),
    (
        "Q_CON_24", "question", "Applied", "BG 2.38", "T04",
        "How can someone perform high-stakes duty without incurring moral guilt or emotional burnout from success and failure?",
        "By treating pleasure and pain, gain and loss, and victory and defeat alike (sama), and engaging in duty without personal attachment to the outcome."
    ),
    (
        "Q_CON_25", "question", "Interpretive", "BG 2.39", "T05",
        "What is the transition Krishna marks between Sankhya and Buddhi-yoga, and what does Buddhi-yoga free a person from?",
        "Sankhya refers to the discriminative knowledge of the eternal Self just taught, while Buddhi-yoga (Karma-yoga performed with equanimous intellect) enables one to cast off the bondage of action (karma-bandham)."
    ),
    (
        "Q_CON_26", "question", "Factual", "BG 2.40", "T05",
        "What two assurances does Krishna give about even a small effort made on the path of Nishkama Karma / Buddhi-yoga?",
        "There is no loss of effort begun (nehabhikrama-nasho'sti) and no adverse result (pratyavayo na vidyate); even a little of this dharma protects from great fear."
    ),
    (
        "Q_CON_27", "question", "Factual", "BG 2.41", "T05",
        "How does Krishna contrast the intellect of a resolute spiritual seeker with the intellect of the irresolute?",
        "The resolute intellect (vyavasayatmika buddhih) is single-pointed (eka), whereas the intellects of the irresolute are many-branched and endless (bahu-shakha hy anantash ca)."
    ),
    (
        "Q_CON_28", "question", "Interpretive", "BG 2.42, BG 2.43, BG 2.44", "T05",
        "What does Krishna mean by 'flowery speech' (pushpitam vacam), and why do people attached to pleasure and power fail to attain samadhi?",
        "Flowery speech refers to ritualistic Vedic promises of heaven, rebirth, wealth, and enjoyment; those whose minds are carried away by such desires lack the one-pointed intellect needed for samadhi."
    ),
    (
        "Q_CON_29", "question", "Factual", "BG 2.45", "T05",
        "What four instructions does Krishna give Arjuna after stating that the Vedas deal with the three gunas (traigunya-vishaya vedah)?",
        "He instructs Arjuna to transcend the three gunas (nistraunya), be free from dualities (nirdvandva), remain ever established in sattva (nitya-sattvastha), be free from anxiety over acquisition and preservation (niryoga-kshema), and be established in the Self (atmavan)."
    ),
    (
        "Q_CON_30", "question", "Interpretive", "BG 2.46", "T05",
        "Explain the analogy of the small water well (udapana) and the vast flooded reservoir (samplutodaka) in relation to the Vedas.",
        "Just as a small well serves only a limited purpose when water is flooded everywhere, the ritualistic rewards of the Vedas are subsumed for an enlightened knower of Brahman (or for one who grasps the supreme spiritual purport of all scriptures)."
    ),
    (
        "Q_CON_31", "question", "Factual", "BG 2.47", "T06",
        "What are the four distinct clauses (catuh-sutri) of Karma Yoga stated in the famous verse on action and fruits?",
        "(1) Your right is to action alone, (2) never to its fruits, (3) do not let the fruits of action be your motive, and (4) never become attached to inaction (akarma)."
    ),
    (
        "Q_CON_32", "question", "Factual", "BG 2.48", "T06",
        "How does Krishna define 'Yoga' in terms of success and failure when instructing Arjuna to perform action?",
        "He defines Yoga as evenness of mind (samatvam yoga ucyate)—remaining equal in success and failure (siddhy-asiddhyoh samo bhutva) while abandoning attachment."
    ),
    (
        "Q_CON_33", "question", "Interpretive", "BG 2.49, BG 2.50", "T06",
        "Why does Krishna call those motivated by selfish results 'wretched' (kripanah), and how do commentaries explain 'yogah karmasu kaushalam'?",
        "Selfish fruitive action is far inferior to Buddhi-yoga, so seekers of paltry fruits are called kripanah; 'yogah karmasu kaushalam' means the skill in action that turns binding work into liberating worship through equanimity or devotion to God."
    ),
    (
        "Q_CON_34", "question", "Factual", "BG 2.51", "T06",
        "What state do wise people endowed with equanimous intellect attain by renouncing the fruits born of action?",
        "Freed from the bondage of rebirth (janma-bandha-vinirmuktah), they reach the sorrowless, flawless state (padam anamayam)."
    ),
    (
        "Q_CON_35", "question", "Factual", "BG 2.52, BG 2.53", "T06",
        "When does a seeker attain true Yoga according to Krishna's description of crossing the mire of delusion (moha-kalila)?",
        "When the intellect crosses beyond the mire of delusion, becomes indifferent to worldly things heard and yet to be heard, and stands unwavering in samadhi undisturbed by conflicting scriptural injunctions."
    ),
    (
        "Q_CON_36", "question", "Factual", "BG 2.54", "T07",
        "What four specific questions does Arjuna ask Krishna to identify a Sthitaprajna (person of steady wisdom)?",
        "Arjuna asks: (1) What is the description/mark (का भाषा) of a Sthitaprajna in samadhi? (2) How does he speak? (3) How does he sit (abide)? and (4) How does he walk (act in the world)?"
    ),
    (
        "Q_CON_37", "question", "Factual", "BG 2.55", "T07",
        "What is Krishna's first criterion for calling someone a Sthitaprajna?",
        "When a person completely casts off all selfish desires entering the mind (prajahati yada kaman) and is content in the Self through the Self (atmany evatmana tushtah)."
    ),
    (
        "Q_CON_38", "question", "Factual", "BG 2.56, BG 2.57", "T07",
        "How does a sage of steady wisdom (sthitadhih munih) react to sorrow, pleasure, fear, anger, and auspicious or inauspicious events?",
        "His mind is unshaken in sorrows, free from craving in pleasures, free from attachment, fear, and anger, and he neither rejoices over good fortune nor hates misfortune."
    ),
    (
        "Q_CON_39", "question", "Factual", "BG 2.58", "T07",
        "How does the simile of the tortoise (kurma) illustrate the sense mastery of a Sthitaprajna?",
        "Just as a tortoise withdraws its limbs on all sides into its shell, the person of steady wisdom withdraws the senses completely from sense objects."
    ),
    (
        "Q_CON_40", "question", "Interpretive", "BG 2.59", "T08",
        "What is the difference between sense objects turning away (vishaya vinivartante) through fasting and the disappearance of inner craving (rasa)?",
        "External sense objects turn away when a person abstains physically (nirahara), but the subtle inner taste or longing (rasa) remains until it too disappears upon experiencing the Supreme (param drishtva)."
    ),
    (
        "Q_CON_41", "question", "Factual", "BG 2.60, BG 2.61", "T08",
        "Why is mere human effort insufficient to control the turbulent senses, and what must a seeker focus on while restraining them?",
        "Even for a wise and striving person, the turbulent senses forcibly carry away the mind; therefore one must restrain them all and sit steadfastly focused on Krishna as the Supreme (mat-parah)."
    ),
    (
        "Q_CON_42", "question", "Cross-verse", "BG 2.62, BG 2.63", "T08",
        "Trace the exact eight-step psychological chain of ruin that begins with brooding over sense objects.",
        "(1) Brooding on sense objects -> (2) attachment (sanga) -> (3) desire (kama) -> (4) anger (krodha) -> (5) delusion (sammoha) -> (6) loss of memory (smriti-vibhrama) -> (7) destruction of intellect (buddhi-nasha) -> (8) complete ruin (pranashyati)."
    ),
    (
        "Q_CON_43", "question", "Applied", "BG 2.62, BG 2.63", "T08",
        "Why does repeated mental dwelling on a temptation eventually destroy a person's judgment even if they initially think they are just looking?",
        "Because mental contemplation (dhyayato vishayan) breeds attachment and desire; when frustrated it turns into anger and delusion, erasing memory of right and wrong and destroying the intellect."
    ),
    (
        "Q_CON_44", "question", "Factual", "BG 2.64, BG 2.65", "T08",
        "How can a person move among sense objects in the world without falling into bondage, and what happens when they attain inner serenity (prasada)?",
        "By moving among sense objects with senses controlled by the Self and free from attraction and aversion (raga-dvesha-vimuktaih), one attains serenity (prasada), in which all sorrows end and the intellect quickly becomes steady."
    ),
    (
        "Q_CON_45", "question", "Factual", "BG 2.66", "T08",
        "Why can an undisciplined person (ayukta) never experience true happiness according to Krishna's chain of negations?",
        "For the undisciplined there is no steady intellect (buddhi) and no meditation/contemplation (bhavana); without contemplation there is no peace (shanti), and without peace how can there be happiness?"
    ),
    (
        "Q_CON_46", "question", "Factual", "BG 2.67, BG 2.68", "T08",
        "What maritime analogy does Krishna use to show how even a single wandering sense can destroy wisdom?",
        "When the mind follows even one of the wandering senses, it carries away a person's wisdom just as a gale sweeps away a boat upon the waters (vayur navam ivambhasi)."
    ),
    (
        "Q_CON_47", "question", "Interpretive", "BG 2.69", "T09",
        "What does the metaphor of 'night for all beings' versus 'night for the seeing sage' mean?",
        "The self-controlled sage is awake to the spiritual reality that is dark as night to ordinary worldly beings, while the transient worldly pursuits in which ordinary beings are wide awake are seen as mere night by the sage."
    ),
    (
        "Q_CON_48", "question", "Factual", "BG 2.70", "T09",
        "How does the analogy of rivers entering the ocean explain why the enlightened sage attains peace while the desirer of desires does not?",
        "Just as waters enter the ocean, which remains full and unmoved in its bed, desires enter the sage without disturbing his inner fullness; thus he attains peace, not the one who chases desires (kama-kami)."
    ),
    (
        "Q_CON_49", "question", "Factual", "BG 2.71, BG 2.72", "T09",
        "What are the three inner renunciations of the person who attains supreme peace, and what is the state called 'Brahmi sthiti'?",
        "Moving free from longing (nihspriha), ownership (nirmama), and egoism (nirahankara), one attains the state of Brahmi sthiti; reaching it, one is never deluded again and attains Brahma-nirvana even at the hour of death."
    ),
    (
        "Q_CON_50", "question", "Cross-verse", "BG 2.55, BG 2.70, BG 2.71", "T09",
        "How does Krishna's portrait of the Sthitaprajna evolve from its opening definition to the closing ocean metaphor?",
        "It begins in 2.55 with casting off mental desires and being content in the Self, and culminates in 2.70-2.71 with the sage remaining unmoved like the ocean while moving in the world free from craving, possessiveness, and ego."
    ),

    # =========================================================================
    # PART 2: 40 PASTED SANSKRIT SHLOKAS WITH NOISY SPELLING (Q_SHL_01 .. Q_SHL_40)
    # =========================================================================
    (
        "Q_SHL_01", "shloka", "Noisy-Shloka", "BG 2.1", "T01",
        "tatha krupayavishtam ashrupurna kulekshanam vishidantam idam vakyam uvacha madhusudanaha",
        "BG 2.1: Sanjaya describes Arjuna overwhelmed by pity and tear-filled eyes, addressed by Madhusudana."
    ),
    (
        "Q_SHL_02", "shloka", "Noisy-Shloka", "BG 2.2", "T01",
        "ಕುತಸ್ತಾ ಕಶ್ಮಲಮಿದಂ ವಿಷಮೇ ಸಮುಪಸ್ಥಿತಂ ಅನಾರ್ಯ ಜುಷ್ಟಮಸ್ವರ್ಗ್ಯಂ ಅಕೀರ್ತಿಕರಮರ್ಜುನ",
        "BG 2.2: Krishna rebukes Arjuna for yielding to delusion (kashmala) at a critical hour."
    ),
    (
        "Q_SHL_03", "shloka", "Noisy-Shloka", "BG 2.3", "T01",
        "klaibyam ma sma gamah partha naitat tvayyupapadyate kshudram hridayadaurbalyam tyaktvottishtha parantapa",
        "BG 2.3: Krishna tells Arjuna not to yield to unmanliness and to cast off petty weakness of heart."
    ),
    (
        "Q_SHL_04", "shloka", "Noisy-Shloka", "BG 2.5", "T01",
        "ಗುರುನ್ ಹತ್ವಾಹಿ ಮಹಾನುಭಾವಾನ್ ಶ್ರೇಯೋ ಭೋಕ್ತುಂ ಭೈಕ್ಷ್ಯಮಪೀಹ ಲೋಕೇ ಹತ್ವಾರ್ಥಕಾಮಾಂಸ್ತು ಗುರುನಿಹೈವ ಭುಂಜೀಯ ಭೋಗಾನ್ ರುಧಿರಪ್ರದಿಗ್ಧಾನ್",
        "BG 2.5: Arjuna says it is better to live on alms than to slay noble teachers and enjoy blood-stained pleasures."
    ),
    (
        "Q_SHL_05", "shloka", "Noisy-Shloka", "BG 2.7", "T01",
        "karpanya doshopahata svabhavah pruchchami tvam dharma sammudha chetah yacchreyah syan nishchitam bruhi tanme shishyas teham shadhi mam tvam prapannam",
        "BG 2.7: Arjuna surrenders to Krishna as a disciple and asks what is truly for his highest good."
    ),
    (
        "Q_SHL_06", "shloka", "Noisy-Shloka", "BG 2.9", "T01",
        "evam uktva hrishikesham gudakeshah parantapa na yotsya iti govindam uktva tushnim babhuva ha",
        "BG 2.9: Having told Hrishikesha 'I will not fight', Arjuna falls silent."
    ),
    (
        "Q_SHL_07", "shloka", "Noisy-Shloka", "BG 2.11", "T02",
        "ಅಶೋಚ್ಯಾನನ್ವಶೋಚಸ್ತ್ವಂ ಪ್ರಜ್ಞಾವಾದಾಂಶ್ಚ ಭಾಷಸೆ ಗತಾಸೂನ್ ಅಗತಾಸೂಶ್ಚ ನಾನುಶೋಚಂತಿ ಪಂಡಿತಾಃ",
        "BG 2.11: Krishna declares that the wise grieve neither for the living nor for the dead."
    ),
    (
        "Q_SHL_08", "shloka", "Noisy-Shloka", "BG 2.12", "T02",
        "natvevaham jatu nasam na tvam neme janadhipah na chaiva na bhavishyamah sarve vayam atah param",
        "BG 2.12: Never was there a time when I, you, or these kings did not exist, nor shall we ever cease to be."
    ),
    (
        "Q_SHL_09", "shloka", "Noisy-Shloka", "BG 2.13", "T02",
        "dehino asmin yatha dehe kaumaram yauvanam jara tatha dehantara praptir dhiras tatra na muhyati",
        "BG 2.13: Just as childhood, youth, and old age happen to the embodied soul, so does acquiring another body."
    ),
    (
        "Q_SHL_10", "shloka", "Noisy-Shloka", "BG 2.14", "T02",
        "matrasparshastu kaunteya shitoshna sukha duhkhadah agamapayino nityas tams titikshasva bharata",
        "BG 2.14: Sensory contacts giving cold, heat, pleasure, and pain come and go; endure them bravely."
    ),
    (
        "Q_SHL_11", "shloka", "Noisy-Shloka", "BG 2.15", "T02",
        "yam hi na vyathayantyete purusham purusharshabha sama duhkha sukham dhiram so amritatvaya kalpate",
        "BG 2.15: The steadfast person whom these contacts do not afflict, equal in pain and pleasure, is fit for immortality."
    ),
    (
        "Q_SHL_12", "shloka", "Noisy-Shloka", "BG 2.16", "T02",
        "ನಾಸತೋ ವಿದ್ಯತೇ ಭಾವೋ ನಾಭಾವೋ ವಿದ್ಯತೇ ಸತಃ ಉಭಯೋರಪಿ ದೃಷ್ಟೋಂತಸ್ತ್ವನಯೋಸ್ತತ್ತ್ವದರ್ಶಿಭಿಃ",
        "BG 2.16: Of the unreal there is no being, and of the real there is no non-being."
    ),
    (
        "Q_SHL_13", "shloka", "Noisy-Shloka", "BG 2.17", "T02",
        "avinashi tu tadviddhi yena sarvam idam tatam vinasham avyayasyasya na kashchit kartum arhati",
        "BG 2.17: Know that to be imperishable by which all this is pervaded; none can destroy the immutable."
    ),
    (
        "Q_SHL_14", "shloka", "Noisy-Shloka", "BG 2.18", "T02",
        "ಅಂತವಂತ ಇಮೇ ದೇಹಾ ನಿತ್ಯಸ್ಯೋಕ್ತಾಃ ಶರೀರಿಣಃ ಅನಾಶಿನೋ ಪ್ರಮೇಯಸ್ಯ ತಸ್ಮಾದ್ಯುಧ್ಯಸ್ವ ಭಾರತ",
        "BG 2.18: These bodies of the eternal, indestructible, immeasurable embodied Self have an end; therefore fight."
    ),
    (
        "Q_SHL_15", "shloka", "Noisy-Shloka", "BG 2.19", "T02",
        "ya enam vetti hantaram yash chainam manyate hatam ubhau tau na vijanito nayam hanti na hanyate",
        "BG 2.19: Neither he who thinks the soul slays nor he who thinks it is slain knows the truth."
    ),
    (
        "Q_SHL_16", "shloka", "Noisy-Shloka", "BG 2.20", "T02",
        "na jayate mriyate va kadachin nayam bhutva bhavita va na bhuyah ajo nityah shashvato yam purano na hanyate hanyamane sharire",
        "BG 2.20: The soul is never born and never dies; unborn, eternal, ancient, it is not slain when the body is slain."
    ),
    (
        "Q_SHL_17", "shloka", "Noisy-Shloka", "BG 2.22", "T02",
        "ವಾಸಾಂಸಿ ಜೀರ್ಣಾನಿ ಯಥಾ ವಿಹಾಯ ನವಾನಿ ಗೃಹ್ಣಾತಿ ನರೋಪರಾಣಿ ತಥಾ ಶರೀರಾಣಿ ವಿಹಾಯ ಜೀರ್ಣಾನ್ಯನ್ಯಾನಿ ಸಂಯಾತಿ ನವಾನಿ ದೇಹೀ",
        "BG 2.22: As a person casts off worn-out garments and puts on new ones, so the embodied soul enters new bodies."
    ),
    (
        "Q_SHL_18", "shloka", "Noisy-Shloka", "BG 2.23", "T02",
        "nainam chindanti shastrani nainam dahati pavakah na chainam kledayantyapo na shoshayati marutah",
        "BG 2.23: Weapons do not cleave the soul, fire does not burn it, water does not wet it, wind does not dry it."
    ),
    (
        "Q_SHL_19", "shloka", "Noisy-Shloka", "BG 2.24", "T02",
        "ಅಚ್ಛೇದ್ಯೋಯಂ ಅದಾಹ್ಯೋಯಂ ಅಕ್ಲೇದ್ಯೋಶೋಷ್ಯ ಏವ ಚ ನಿತ್ಯಃ ಸರ್ವಗತಃ ಸ್ಥಾಣುರಚಲೋಯಂ ಸನಾತನಃ",
        "BG 2.24: The soul is uncleavable, incombustible, insoluble, and eternal, all-pervading, immovable."
    ),
    (
        "Q_SHL_20", "shloka", "Noisy-Shloka", "BG 2.27", "T03",
        "jatasya hi dhruvo mrityur dhruvam janma mritasya cha tasmad apariharye arthe na tvam shochitum arhasi",
        "BG 2.27: For the born death is certain, and for the dead birth is certain; do not grieve over the inevitable."
    ),
    (
        "Q_SHL_21", "shloka", "Noisy-Shloka", "BG 2.28", "T03",
        "avyaktadini bhutani vyaktamadhyani bharata avyaktanidhananyeva tatra ka paridevana",
        "BG 2.28: Beings are unmanifest in the beginning, manifest in the middle, and unmanifest again at the end."
    ),
    (
        "Q_SHL_22", "shloka", "Noisy-Shloka", "BG 2.29", "T03",
        "ashcharyavat pashyati kashchid enam ashcharyavad vadati tathaiva chanyah ashcharyavac chainam anyah shrinoti shrutvapyenam veda na chaiva kashchit",
        "BG 2.29: One sees the soul as a wonder, another speaks of it as a wonder, yet scarcely anyone knows it."
    ),
    (
        "Q_SHL_23", "shloka", "Noisy-Shloka", "BG 2.31", "T04",
        "ಸ್ವಧರ್ಮಮಪಿ ಚಾವೇಕ್ಷ್ಯ ನ ವಿಕಂಪಿತುಮರ್ಹಸಿ ಧರ್ಮ್ಯಾದ್ಧಿ ಯುದ್ಧಾಚ್ಛ್ರೇಯೋನ್ಯತ್ ಕ್ಷತ್ರಿಯಸ್ಯ ನ ವಿದ್ಯತೇ",
        "BG 2.31: Looking to your own svadharma you should not waver; nothing is higher for a Kshatriya than righteous war."
    ),
    (
        "Q_SHL_24", "shloka", "Noisy-Shloka", "BG 2.32", "T04",
        "yadricchaya chopapannam svargadvaram apavritam sukhinah kshatriyah partha labhante yuddham idrisham",
        "BG 2.32: Fortunate Kshatriyas obtain such a battle that comes unsought as an open door to heaven."
    ),
    (
        "Q_SHL_25", "shloka", "Noisy-Shloka", "BG 2.37", "T04",
        "ಹತೋ ವಾ ಪ್ರಾಪ್ಸ್ಯಸಿ ಸ್ವರ್ಗಂ ಜಿತ್ವಾ ವಾ ಭೋಕ್ಷ್ಯಸೇ ಮಹೀಂ ತಸ್ಮಾದುತ್ತಿಷ್ಠ ಕೌಂತೇಯ ಯುದ್ಧಾಯ ಕೃತನಿಶ್ಚಯಃ",
        "BG 2.37: Slain you will gain heaven, victorious you will enjoy the earth; therefore rise resolved to fight."
    ),
    (
        "Q_SHL_26", "shloka", "Noisy-Shloka", "BG 2.38", "T04",
        "sukha duhkhe same kritva labhalabhau jayajayau tato yuddhaya yujyasva naivam papam avapsyasi",
        "BG 2.38: Treating pleasure and pain, gain and loss, victory and defeat alike, engage in battle and incur no sin."
    ),
    (
        "Q_SHL_27", "shloka", "Noisy-Shloka", "BG 2.40", "T05",
        "nehabhikrama nashosti pratyavayo na vidyate svalpam apyasya dharmasya trayate mahato bhayat",
        "BG 2.40: In this path no effort is lost and there is no adverse effect; even a little of this dharma saves from great fear."
    ),
    (
        "Q_SHL_28", "shloka", "Noisy-Shloka", "BG 2.41", "T05",
        "vyavasayatmika buddhir ekeha kurunandana bahushakha hy anantash cha buddhayo avyavasayinam",
        "BG 2.41: The resolute intellect is single-pointed here, whereas the thoughts of the irresolute are many-branched."
    ),
    (
        "Q_SHL_29", "shloka", "Noisy-Shloka", "BG 2.45", "T05",
        "ತ್ರೈಗುಣ್ಯ ವಿಷಯಾ ವೇದಾ ನಿಸ್ತ್ರೈಗುಣ್ಯೋ ಭವಾರ್ಜುನ ನಿರ್ದ್ವಂದ್ವೋ ನಿತ್ಯಸತ್ತ್ವಸ್ಥೋ ನಿರ್ಯೋಗಕ್ಷೇಮ ಆತ್ಮವಾನ್",
        "BG 2.45: The Vedas deal with the three gunas; become free from the three gunas and dualities, O Arjuna."
    ),
    (
        "Q_SHL_30", "shloka", "Noisy-Shloka", "BG 2.46", "T05",
        "yavan artha udapane sarvatah samplutodake tavan sarveshu vedeshu brahmanasya vijanatah",
        "BG 2.46: As much use as there is in a small well when water is flooded everywhere, so much is there in all the Vedas for a knower of Brahman."
    ),
    (
        "Q_SHL_31", "shloka", "Noisy-Shloka", "BG 2.47", "T06",
        "karmanyevadhikaraste ma phaleshu kadachana ma karmaphalaheturbhur ma te sangostvakarmani",
        "BG 2.47: Your right is to action alone, never to the fruits; do not be attached to inaction."
    ),
    (
        "Q_SHL_32", "shloka", "Noisy-Shloka", "BG 2.48", "T06",
        "ಯೋಗಸ್ಥಃ ಕುರು ಕರ್ಮಾಣಿ ಸಂಗಂ ತ್ಯಕ್ತ್ವಾ ಧನಂಜಯ ಸಿದ್ಮ್ಯಸಿದ್ಧ್ಯೋಃ ಸಮೋ ಭೂತ್ವಾ ಸಮತ್ವಂ ಯೋಗ ಉಚ್ಯತೇ",
        "BG 2.48: Perform actions established in Yoga, abandoning attachment and remaining equal in success and failure."
    ),
    (
        "Q_SHL_33", "shloka", "Noisy-Shloka", "BG 2.50", "T06",
        "buddhiyukto jahatiha ubhe sukrita dushkrite tasmad yogaya yujyasva yogah karmasu kaushalam",
        "BG 2.50: Endowed with equanimous intellect one casts off both good and bad reactions; Yoga is skill in action."
    ),
    (
        "Q_SHL_34", "shloka", "Noisy-Shloka", "BG 2.54", "T07",
        "ಸ್ಥಿತಪ್ರಜ್ಞಸ್ಯ ಕಾ ಭಾಷಾ ಸಮಾಧಿಸ್ಥಸ್ಯ ಕೇಶವ ಸ್ಥಿತಧೀಃ ಕಿಂ ಪ್ರಭಾಷೇತ ಕಿಮಾಸೀತ ವ್ರಜೇತ ಕಿಮ್",
        "BG 2.54: Arjuna asks how a Sthitaprajna speaks, sits, and walks."
    ),
    (
        "Q_SHL_35", "shloka", "Noisy-Shloka", "BG 2.55", "T07",
        "prajahati yada kaman sarvan partha manogatan atmanyevatmana tushtah sthitaprajnas tadocyate",
        "BG 2.55: When one casts off all desires of the mind and is satisfied in the Self by the Self, he is called Sthitaprajna."
    ),
    (
        "Q_SHL_36", "shloka", "Noisy-Shloka", "BG 2.56", "T07",
        "ದುಃಖೇಷ್ವನುದ್ವಿಗ್ನಮನಾಃ ಸುಖೇಷು ವಿಗತಸ್ಪೃಹಃ ವೀತರಾಗಭಯಕ್ರೋಧಃ ಸ್ಥಿತಧೀರ್ಮುನಿರುಚ್ಯತೇ",
        "BG 2.56: Undisturbed in sorrow, free from craving in pleasure, free from attachment, fear, and anger—he is a sage of steady wisdom."
    ),
    (
        "Q_SHL_37", "shloka", "Noisy-Shloka", "BG 2.58", "T07",
        "yada samharate chayam kurmo anganiva sarvashah indriyanindriyarthebhyas tasya prajna pratishthita",
        "BG 2.58: When, like a tortoise withdrawing its limbs on all sides, he withdraws the senses from sense objects, his wisdom is steady."
    ),
    (
        "Q_SHL_38", "shloka", "Noisy-Shloka", "BG 2.62, BG 2.63", "T08",
        "ಧ್ಯಾಯತೋ ವಿಷಯಾನ್ ಪುಂಸಃ ಸಂಗಸ್ತೇಷೂಪಜಾಯತೇ ಸಂಗಾತ್ ಸಂಜಾಯತೇ ಕಾಮಃ ಕಾಮಾತ್ ಕ್ರೋಧೋಭಿಜಾಯತೇ ಕ್ರೋಧಾದ್ ಭವತಿ ಸಮ್ಮೋಹಃ",
        "BG 2.62-2.63: Dwelling on sense objects breeds attachment, desire, anger, delusion, loss of memory, and destruction of intellect."
    ),
    (
        "Q_SHL_39", "shloka", "Noisy-Shloka", "BG 2.69", "T09",
        "ya nisha sarvabhutanam tasyam jagarti samyami yasyam jagrati bhutani sa nisha pashyato muneh",
        "BG 2.69: What is night for all beings, in that the self-controlled sage is awake."
    ),
    (
        "Q_SHL_40", "shloka", "Noisy-Shloka", "BG 2.70", "T09",
        "apuryamanam achalapratishtham samudram apah pravishanti yadvat tadvat kama yam pravishanti sarve sa shantim apnoti na kamakami",
        "BG 2.70: As waters enter the full and unmoved ocean, he whom desires enter without disturbing attains peace, not the desirer of desires."
    ),

    # =========================================================================
    # PART 3: 35 PASTED EXPLANATION EXCERPTS (Q_EXP_01 .. Q_EXP_35)
    # =========================================================================
    (
        "Q_EXP_01", "explanation", "Explanation-Excerpt", "BG 2.1, BG 2.2", "T01",
        "Overcome by sudden faintheartedness while looking at his kinsmen, the warrior's eyes brimmed with tears of despair. The Lord immediately reproached this mental cloudiness as unworthy of noble souls, closing the gates of heaven and bringing disgrace at a decisive moment.",
        "Explanation excerpt grounding to BG 2.1-2.2 (Arjuna's tearful breakdown and Krishna's rebuke of kashmala)."
    ),
    (
        "Q_EXP_02", "explanation", "Explanation-Excerpt", "BG 2.4, BG 2.5", "T01",
        "How can a disciple loose lethal shafts against revered preceptors and grandsires who deserve worship with flowers rather than weapons? Even wandering as a mendicant eating alms would be nobler than feasting on royal luxuries stained with the blood of one's own teachers.",
        "Explanation excerpt grounding to BG 2.4-2.5 (Arjuna's refusal to strike Bhishma and Drona and preference for begging)."
    ),
    (
        "Q_EXP_03", "explanation", "Explanation-Excerpt", "BG 2.7, BG 2.8", "T01",
        "Recognizing that his own judgment is paralyzed by weak-heartedness and moral perplexity over duty, the hero drops his arguments and surrenders completely as a student at the Master's feet, confessing that even unrivaled sovereignty over earth and the celestials could not dry up the grief scorching his senses.",
        "Explanation excerpt grounding to BG 2.7-2.8 (Arjuna's sharanagati as disciple and inconsolable grief)."
    ),
    (
        "Q_EXP_04", "explanation", "Explanation-Excerpt", "BG 2.11", "T02",
        "Here the true spiritual instruction begins: although you speak words that sound like high philosophy, you are mourning for those who deserve no grief. Men of true knowledge never lament either for bodies from which life has departed or for those in which breath still remains.",
        "Explanation excerpt grounding to BG 2.11 (ashocyan anvashocas tvam—the wise grieve neither for the living nor the dead)."
    ),
    (
        "Q_EXP_05", "explanation", "Explanation-Excerpt", "BG 2.12, BG 2.13", "T02",
        "Look at a childhood photograph and compare it with youth and old age: the physical frame has changed completely, yet the conscious person recognizing 'this was I' remains continuous. In the very same way, death is merely the soul stepping into another bodily frame, and the wise are never deluded by it.",
        "Explanation excerpt grounding to BG 2.12-2.13 (continuity of the eternal conscious self across childhood, youth, old age, and death)."
    ),
    (
        "Q_EXP_06", "explanation", "Explanation-Excerpt", "BG 2.14, BG 2.15", "T02",
        "When the five senses touch external objects, they produce fleeting waves of cold and heat, joy and sorrow. Because these experiences have a beginning and an end, they are impermanent by nature; the courageous seeker who endures them with equal vision becomes worthy of liberation.",
        "Explanation excerpt grounding to BG 2.14-2.15 (matra-sparsha, titiksha, and equanimity leading to immortality)."
    ),
    (
        "Q_EXP_07", "explanation", "Explanation-Excerpt", "BG 2.16, BG 2.17", "T02",
        "Seers of truth who have examined both the transient modifications of matter and the unchanging substratum conclude that what is unreal never acquires true permanence, and what is real never ceases to exist. That imperishable reality pervades the entire cosmos, and no power can ever annihilate it.",
        "Explanation excerpt grounding to BG 2.16-2.17 (sat vs asat and the all-pervading imperishable reality)."
    ),
    (
        "Q_EXP_08", "explanation", "Explanation-Excerpt", "BG 2.18, BG 2.19", "T02",
        "Only the material envelopes are subject to termination, whereas the indwelling owner of the body is immeasurable and deathless. Therefore, anyone who imagines that the inner Self kills another or can be killed by weapons is completely ignorant of spiritual reality.",
        "Explanation excerpt grounding to BG 2.18-2.19 (mortal bodies vs immortal sharirin that neither slays nor is slain)."
    ),
    (
        "Q_EXP_09", "explanation", "Explanation-Excerpt", "BG 2.20, BG 2.21", "T02",
        "Free from the six modifications that afflict all material things, the conscious entity has no birth, growth, decay, or death; it is ancient, unborn, and changeless even when its physical dwelling is struck down. Knowing this immutable nature, how can an enlightened person believe anyone is truly slain?",
        "Explanation excerpt grounding to BG 2.20-2.21 (unborn, eternal Atman unaffected when the body is slain)."
    ),
    (
        "Q_EXP_10", "explanation", "Explanation-Excerpt", "BG 2.22", "T02",
        "Just as a person discards threadbare, torn clothes at the end of the day and puts on fresh attire without weeping over the old fabric, the indwelling soul casts aside decrepit physical bodies and enters new ones appropriate for its onward journey.",
        "Explanation excerpt grounding to BG 2.22 (the worn-out garments analogy for transmigration)."
    ),
    (
        "Q_EXP_11", "explanation", "Explanation-Excerpt", "BG 2.23, BG 2.24, BG 2.25", "T02",
        "Because the Self has no physical parts orlimbs, sharp swords cannot cut it, blazing fire cannot reduce it to ashes, water cannot dissolve it, and dry winds cannot wither it. Being all-pervading, fixed, unmanifest to the eyes, and beyond mental fabrication, it gives no ground for lamentation.",
        "Explanation excerpt grounding to BG 2.23-2.25 (the four elements cannot harm the unmanifest, inconceivable Atman)."
    ),
    (
        "Q_EXP_12", "explanation", "Explanation-Excerpt", "BG 2.26, BG 2.27", "T03",
        "Even if we set aside higher metaphysics and adopt the common worldly view that the self dies with the body, grief is still futile. Whatever takes birth must inevitably die, and what dies must take birth again; mourning over an unavoidable law of nature helps no one.",
        "Explanation excerpt grounding to BG 2.26-2.27 (inevitability of death for the born and rebirth for the dead)."
    ),
    (
        "Q_EXP_13", "explanation", "Explanation-Excerpt", "BG 2.28, BG 2.29, BG 2.30", "T03",
        "All creatures emerge from an unmanifest state before birth, appear briefly on the visible stage in the middle, and vanish back into the unmanifest at death—like figures seen in a dream or travelers meeting briefly. Rare is the sage who truly comprehends this wondrous indweller who remains forever invulnerable in every body.",
        "Explanation excerpt grounding to BG 2.28-2.30 (unmanifest beginning and end, the wondrous soul, and summary against grief)."
    ),
    (
        "Q_EXP_14", "explanation", "Explanation-Excerpt", "BG 2.31, BG 2.32", "T04",
        "Turning from metaphysics to the ethics of one's own station in life, a protector of society must never tremble before a righteous struggle. Such a battle for justice arrives unsought like the wide-open gates of heaven, and only fortunate warriors receive the privilege to fight it.",
        "Explanation excerpt grounding to BG 2.31-2.32 (svadharma of a Kshatriya and righteous war as an open door to heaven)."
    ),
    (
        "Q_EXP_15", "explanation", "Explanation-Excerpt", "BG 2.33, BG 2.34", "T04",
        "If a guardian of dharma turns his back on righteous duty out of personal sentiment, he forfeits both virtue and honor, incurring grave transgression. The world will recount his everlasting infamy, and for a person who has enjoyed high esteem, disgrace is far more painful than physical death.",
        "Explanation excerpt grounding to BG 2.33-2.34 (sin of abandoning svadharma and dishonor being worse than death)."
    ),
    (
        "Q_EXP_16", "explanation", "Explanation-Excerpt", "BG 2.35, BG 2.36", "T04",
        "The mighty commanders who once held you in high regard will never believe you left the field out of compassion; they will conclude that you fled in terror, and your adversaries will mock your prowess with cruel, unutterable taunts.",
        "Explanation excerpt grounding to BG 2.35-2.36 (maharathas thinking Arjuna fled in fear and enemies slandering his skill)."
    ),
    (
        "Q_EXP_17", "explanation", "Explanation-Excerpt", "BG 2.37, BG 2.38", "T04",
        "Looked at either way, righteous battle holds no loss: if you fall on the field you ascend to celestial realms, and if you triumph you rule the earth. Therefore stand up with firm resolve, treating pleasure and pain, profit and loss, and victory and defeat as equal so that no sin touches you.",
        "Explanation excerpt grounding to BG 2.37-2.38 (heaven if slain, earth if victorious, and fighting with equanimity)."
    ),
    (
        "Q_EXP_18", "explanation", "Explanation-Excerpt", "BG 2.39, BG 2.40", "T05",
        "Having imparted the analytical wisdom regarding the Self, the Lord now reveals the discipline of equanimous action by which one breaks the fetters of karma. On this spiritual path no sincere beginning is ever wasted, no partial effort produces harm, and even a small practice rescues the seeker from the terror of samsara.",
        "Explanation excerpt grounding to BG 2.39-2.40 (transition from Sankhya to Buddhi-yoga and assurance that no effort is lost)."
    ),
    (
        "Q_EXP_19", "explanation", "Explanation-Excerpt", "BG 2.41", "T05",
        "A true spiritual aspirant possesses a single, resolute conviction centered on liberation and pleasing the Supreme Lord, whereas those driven by worldly cravings let their intellects splinter into endless branches of competing desires.",
        "Explanation excerpt grounding to BG 2.41 (vyavasayatmika buddhi vs the many-branched minds of the irresolute)."
    ),
    (
        "Q_EXP_20", "explanation", "Explanation-Excerpt", "BG 2.42, BG 2.43, BG 2.44", "T05",
        "Unwise people enamored of ornate ritualistic promises declare that nothing exists beyond heavenly pleasures, power, and opulent rebirth. Like a foolish farmer who plants rice only for the straw and throws away the grain, those whose minds are stolen by luxury cannot attain a steady intellect in samadhi.",
        "Explanation excerpt grounding to BG 2.42-2.44 (critique of flowery ritualistic speech and attachment to pleasure and power)."
    ),
    (
        "Q_EXP_21", "explanation", "Explanation-Excerpt", "BG 2.45, BG 2.46", "T05",
        "The ritual sections of scripture operate within the three modes of material nature, so the seeker must rise above those three modes, free from dualities and anxiety over gain and safety. When the land is flooded with boundless water on all sides, what need remains for a tiny village well?",
        "Explanation excerpt grounding to BG 2.45-2.46 (rising above the three gunas and the small well vs flooded reservoir analogy)."
    ),
    (
        "Q_EXP_22", "explanation", "Explanation-Excerpt", "BG 2.47", "T06",
        "Human authority extends strictly to the performance of prescribed duty, never to the control or ownership of its eventual results. One must neither work as a mercenary driven by the reward nor slip into lazy passivity out of reluctance to act.",
        "Explanation excerpt grounding to BG 2.47 (karmany evadhikaras te—right to action alone, never to fruits or inaction)."
    ),
    (
        "Q_EXP_23", "explanation", "Explanation-Excerpt", "BG 2.48, BG 2.49, BG 2.50", "T06",
        "Just as a traveler who drops his personal claim over imported goods pays no customs tax at the border, the yogi who acts with an even mind in success and failure casts off both merit and demerit right here in this world. That inner poised dedication is the true skill in action.",
        "Explanation excerpt grounding to BG 2.48-2.50 (samatvam yoga ucyate and yogah karmasu kaushalam)."
    ),
    (
        "Q_EXP_24", "explanation", "Explanation-Excerpt", "BG 2.51, BG 2.52, BG 2.53", "T06",
        "When the intellect finally crosses beyond the turbid swamp of delusion, the seeker becomes disenchanted with all worldly rewards heard in the past or promised in the future. Standing unshaken in meditation despite conflicting theological opinions, the wise attain freedom from rebirth and reach the sorrowless abode.",
        "Explanation excerpt grounding to BG 2.51-2.53 (crossing moha-kalila, nirveda, and unwavering samadhi)."
    ),
    (
        "Q_EXP_25", "explanation", "Explanation-Excerpt", "BG 2.54, BG 2.55", "T07",
        "Prompted by the question of how an enlightened sage speaks, sits, and moves through daily life, the Lord replies that the primary mark is purely inward: when a person relinquishes all cravings fabricated by the mind and rejoices in the Self alone, he is established in steady wisdom.",
        "Explanation excerpt grounding to BG 2.54-2.55 (Arjuna's question on the marks of a Sthitaprajna and inward contentment)."
    ),
    (
        "Q_EXP_26", "explanation", "Explanation-Excerpt", "BG 2.56, BG 2.57", "T07",
        "Adversity does not agitate his heart, prosperity awakens no greedy longing, and the three storms of infatuation, fear, and wrath have subsided. Meeting pleasant or unpleasant circumstances without elation or resentment, such a contemplative sage possesses unshakable wisdom.",
        "Explanation excerpt grounding to BG 2.56-2.57 (equanimity of the sthitadhih muni in sorrow, pleasure, and good/bad fortune)."
    ),
    (
        "Q_EXP_27", "explanation", "Explanation-Excerpt", "BG 2.58", "T07",
        "Consider how a tortoise instantly pulls its head and four legs beneath its hard shell the moment danger approaches, exposing nothing to harm; in the same manner, the wise practitioner draws every sense organ inward away from tempting sense objects.",
        "Explanation excerpt grounding to BG 2.58 (the tortoise withdrawing its limbs simile for sense withdrawal)."
    ),
    (
        "Q_EXP_28", "explanation", "Explanation-Excerpt", "BG 2.59", "T08",
        "A sick person or someone observing a strict fast may physically stay away from food and sensory indulgences, yet the subtle mental relish for those objects lingers inside. That deep-seated craving vanishes completely only when one beholds the supreme beauty of the Divine.",
        "Explanation excerpt grounding to BG 2.59 (vishaya vinivartante niraharasya dehinah rasavarjam—fasting vs seeing the Supreme)."
    ),
    (
        "Q_EXP_29", "explanation", "Explanation-Excerpt", "BG 2.60, BG 2.61", "T08",
        "The senses are so impetuous that they can drag away the mind even of a learned scholar who is actively striving for self-control. Therefore, rather than relying on willpower alone, one must yoke the senses and fix the heart devoutly upon the Supreme Lord.",
        "Explanation excerpt grounding to BG 2.60-2.61 (turbulent senses dragging the wise mind and fixing consciousness on Krishna)."
    ),
    (
        "Q_EXP_30", "explanation", "Explanation-Excerpt", "BG 2.62, BG 2.63", "T08",
        "Downfall begins quietly in the mind: repeated contemplation of sensory attractions gives birth to attachment, which ripens into craving; when craving is thwarted it erupts as anger, clouding judgment with delusion, erasing memory of one's spiritual identity, ruining the intellect, and plunging the person into destruction.",
        "Explanation excerpt grounding to BG 2.62-2.63 (the causal cascade from brooding on sense objects to complete ruin)."
    ),
    (
        "Q_EXP_31", "explanation", "Explanation-Excerpt", "BG 2.64, BG 2.65", "T08",
        "The solution is not to become an inert stone, but to move through the world with senses governed by a disciplined mind purged of attraction and hatred. Such a person attains pellucid inner tranquility, in which all mental miseries melt away and the intellect stands firm.",
        "Explanation excerpt grounding to BG 2.64-2.65 (moving among sense objects free from raga-dvesha and attaining prasada)."
    ),
    (
        "Q_EXP_32", "explanation", "Explanation-Excerpt", "BG 2.66, BG 2.67, BG 2.68", "T08",
        "Without self-discipline there is neither decisive intellect nor meditative focus, and without peace how can anyone find happiness? If the mind surrenders to even a single roaming sense, its wisdom is capsized just as a fierce wind blows an unanchored ship off course across the waters.",
        "Explanation excerpt grounding to BG 2.66-2.68 (no peace or happiness for the undisciplined, and the boat swept by wind analogy)."
    ),
    (
        "Q_EXP_33", "explanation", "Explanation-Excerpt", "BG 2.69", "T09",
        "Ordinary people and the enlightened sage live in opposite worlds of awareness: the spiritual reality in which the self-controlled yogi is wide awake seems like dark night to the ignorant, whereas the feverish pursuit of sense gratification in which the world stays awake is regarded as night by the seeing sage.",
        "Explanation excerpt grounding to BG 2.69 (ya nisha sarvabhutanam—what is night for all beings is day for the sage)."
    ),
    (
        "Q_EXP_34", "explanation", "Explanation-Excerpt", "BG 2.70", "T09",
        "Countless swollen rivers pour their waters into the vast ocean every day, yet the ocean neither overflows its banks nor loses its deep stillness. In the same way, sensory experiences and desires may flow into the sage without disturbing his inner equilibrium, granting him lasting peace.",
        "Explanation excerpt grounding to BG 2.70 (the ocean remaining unmoved as rivers enter it)."
    ),
    (
        "Q_EXP_35", "explanation", "Explanation-Excerpt", "BG 2.71, BG 2.72", "T09",
        "Walking through life after shedding all selfish cravings, devoid of the sense of 'mine' and the false ego of 'I', the sage abides in the divine state of Brahman. Having reached this supreme state, one never falls back into delusion and attains final liberation even at the moment of death.",
        "Explanation excerpt grounding to BG 2.71-2.72 (nirmamo nirahankarah and Brahmi sthiti leading to Brahma-nirvana)."
    ),

    # =========================================================================
    # PART 4: 25 UNANSWERABLE / OUT-OF-SCOPE QUERIES (Q_OOS_01 .. Q_OOS_25)
    # =========================================================================
    (
        "Q_OOS_01", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What are the exact names of the conches blown by Krishna, Arjuna, Bhima, Yudhishthira, Nakula, and Sahadeva before the battle begins?",
        "ABSTAIN: Conch names (Panchajanya, Devadatta, Paundra, Anantavijaya, Sughosha, Manipushpaka) belong to Bhagavad Gita Chapter 1 (1.15-1.16), not Chapter 2."
    ),
    (
        "Q_OOS_02", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "How does Krishna explain the creation of the four varnas (chatur-varnyam maya srishtam) based on guna and karma?",
        "ABSTAIN: Chatur-varnya creation is taught in Bhagavad Gita Chapter 4 (4.13), which is outside the Chapter 2 lecture corpus."
    ),
    (
        "Q_OOS_03", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What are the four types of pious people (arto jijnasur artharthi jnani) who worship Krishna, and why is the jnani the best?",
        "ABSTAIN: The four types of devotees are discussed in Bhagavad Gita Chapter 7 (7.16-7.17), not in Chapter 2."
    ),
    (
        "Q_OOS_04", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "How does Krishna describe the offering of a leaf, a flower, a fruit, or water (patram pushpam phalam toyam) with devotion?",
        "ABSTAIN: Patram pushpam phalam toyam is Bhagavad Gita Chapter 9 (9.26), which is not covered in the Chapter 2 lectures."
    ),
    (
        "Q_OOS_05", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What are the main vibhutis (divine opulences) Krishna lists when declaring that among mountains He is Meru and among rivers He is the Ganga?",
        "ABSTAIN: The Vibhuti-yoga catalogue belongs to Bhagavad Gita Chapter 10, outside the Chapter 2 corpus."
    ),
    (
        "Q_OOS_06", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "Describe the terrifying cosmic form (Vishvarupa) with countless mouths and blazing fires in which Arjuna sees the warriors rushing to their destruction.",
        "ABSTAIN: The Vishvarupa-darshana occurs in Bhagavad Gita Chapter 11, not in Chapter 2."
    ),
    (
        "Q_OOS_07", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What are the thirty-six qualities of a devotee dear to Krishna (bhaktiman yah sa me priyah) taught in the chapter on Bhakti Yoga?",
        "ABSTAIN: The marks of the dear devotee belong to Bhagavad Gita Chapter 12 (12.13-12.20), not Chapter 2."
    ),
    (
        "Q_OOS_08", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "How does Krishna define the Kshetra (the field) and the Kshetrajna (the knower of the field) along with the twenty-four cosmic elements?",
        "ABSTAIN: Kshetra-kshetrajna-vibhaga belongs to Bhagavad Gita Chapter 13, outside Chapter 2."
    ),
    (
        "Q_OOS_09", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "Explain the metaphor of the inverted banyan tree (urdhva-mulam adhah-shakham ashvattham) whose leaves are the Vedic hymns.",
        "ABSTAIN: The inverted ashvattha tree is taught in Bhagavad Gita Chapter 15 (15.1-15.3), not in Chapter 2."
    ),
    (
        "Q_OOS_10", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "How does Krishna classify food, sacrifice, austerity, and charity into sattvika, rajasika, and tamasika categories?",
        "ABSTAIN: The three-guna classification of food, yajna, tapas, and dana is in Bhagavad Gita Chapter 17, not Chapter 2."
    ),
    (
        "Q_OOS_11", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What are the five causal factors (adhishthanam tatha karta karanam cha prithag-vidham) required for the accomplishment of any action?",
        "ABSTAIN: The five factors of action are taught in Bhagavad Gita Chapter 18 (18.13-18.15), outside Chapter 2."
    ),
    (
        "Q_OOS_12", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What are the three specific boons that young Nachiketa asks from Yama Dharmaraja in the Katha Upanishad?",
        "ABSTAIN: Nachiketa's three boons in the Katha Upanishad are not taught in the Bhagavad Gita Chapter 2 lectures."
    ),
    (
        "Q_OOS_13", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "How did Hiranyakashipu attempt to kill Prahlada using elephants, serpents, and Holika before Lord Narasimha appeared from the pillar?",
        "ABSTAIN: The detailed narrative of Hiranyakashipu's tortures of Prahlada from Srimad Bhagavatam Canto 7 is outside the scope of Gita Chapter 2."
    ),
    (
        "Q_OOS_14", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What is the structure of the four adhyayas and sixteen padas of Badarayana's Brahma Sutras?",
        "ABSTAIN: The structural organization of the Brahma Sutras is not covered in the Chapter 2 lecture corpus."
    ),
    (
        "Q_OOS_15", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "According to Patanjali's Yoga Sutras, what are the five yamas and five niyamas in the eight limbs of Ashtanga Yoga?",
        "ABSTAIN: Patanjali's Ashtanga Yoga yama/niyama taxonomy is not covered in the Bhagavad Gita Chapter 2 lectures."
    ),
    (
        "Q_OOS_16", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "Why did Kaikeyi demand Bharata's coronation and Rama's fourteen-year exile to the Dandaka forest in the Ayodhya Kanda?",
        "ABSTAIN: Ramayana Ayodhya Kanda narrative events are outside the Bhagavad Gita Chapter 2 corpus."
    ),
    (
        "Q_OOS_17", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What are the exact astronomical planetary positions and calendar year in which the Mahabharata war took place according to archaeoastronomy?",
        "ABSTAIN: Archaeoastronomical dating of the Kurukshetra war is not discussed in the Chapter 2 lectures."
    ),
    (
        "Q_OOS_18", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What are the Vedic rules for performing daily Trikala Sandhyavandanam, pranayama counts, and arghya pradanam?",
        "ABSTAIN: Liturgical rules for Sandhyavandanam and arghya are not covered in the Chapter 2 lectures."
    ),
    (
        "Q_OOS_19", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "Which grains and vegetables are strictly prohibited on Ekadashi fasting days according to Dharmashastra texts?",
        "ABSTAIN: Dietary rules for Ekadashi observance are out of scope for the Chapter 2 lectures."
    ),
    (
        "Q_OOS_20", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "How does Ayurveda classify bodily diseases according to the imbalance of Vata, Pitta, and Kapha doshas?",
        "ABSTAIN: Ayurvedic tridosha pathology (Vata, Pitta, Kapha) is not discussed in the Bhagavad Gita Chapter 2 lectures."
    ),
    (
        "Q_OOS_21", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What are the architectural Agama rules for constructing a temple garbhagriha and installing a stone murti (prana-pratishtha)?",
        "ABSTAIN: Temple Agama architecture and prana-pratishtha rituals are outside the Chapter 2 lecture corpus."
    ),
    (
        "Q_OOS_22", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What is the chronological succession of pontiffs (peethadhipatis) of the Sringeri, Udupi, and Kanchi mathas in the twentieth century?",
        "ABSTAIN: Modern institutional histories of monastic mathas are not in the Chapter 2 lectures."
    ),
    (
        "Q_OOS_23", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "How should an investor balance cryptocurrency volatility and stock-market options trading according to Chapter 2 of the Gita?",
        "ABSTAIN: Financial trading strategies and cryptocurrency investments are out of scope and unsupported by the lectures."
    ),
    (
        "Q_OOS_24", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "How do transformer attention heads and BGE-M3 dense vector embeddings tokenize Sanskrit compound words in Python?",
        "ABSTAIN: Machine learning and NLP implementation details are not discussed in the Bhagavad Gita lectures."
    ),
    (
        "Q_OOS_25", "unanswerable", "Out-of-Scope", "NONE", "OOS",
        "What were the birthplaces, parents' names, and childhood biographies of Shankara, Ramanuja, and Madhvacharya?",
        "ABSTAIN: Biographical histories of the three Acharyas' childhoods are not covered in the Chapter 2 verse lectures."
    ),
]


def main():
    assert len(QUERIES_150) == 150, f"Expected 150 queries, got {len(QUERIES_150)}"
    modes = Counter(q[1] for q in QUERIES_150)
    assert modes == {
        "question": 50,
        "shloka": 40,
        "explanation": 35,
        "unanswerable": 25,
    }, f"Mode counts mismatch: {modes}"

    rows = []
    for qid, mode, qtype, exp_verse, theme_id, text_no_ref, exp_ans in QUERIES_150:
        if exp_verse != "NONE":
            question_en = f"{text_no_ref} ({exp_verse})"
        else:
            question_en = text_no_ref
        rows.append(
            {
                "question_id": qid,
                "input_mode": mode,
                "question_type": qtype,
                "theme_id": theme_id,
                "expected_verse": exp_verse,
                "question_en": question_en,
                "question_en_no_ref": text_no_ref,
                "question_kn": "",
                "expected_answer_en": exp_ans,
            }
        )

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = [
        "question_id",
        "input_mode",
        "question_type",
        "theme_id",
        "expected_verse",
        "question_en",
        "question_en_no_ref",
        "question_kn",
        "expected_answer_en",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} queries -> {OUT_CSV}")
    print("Breakdown by input_mode:", dict(modes))
    print("Breakdown by theme_id  :", dict(Counter(r["theme_id"] for r in rows)))

    try:
        import pandas as pd

        pd.DataFrame(rows).to_excel(OUT_XLSX, index=False)
        print(f"Wrote Excel file       -> {OUT_XLSX}")
    except Exception as exc:
        print(f"(Skipped .xlsx write: {exc}; CSV is ready)")


if __name__ == "__main__":
    main()