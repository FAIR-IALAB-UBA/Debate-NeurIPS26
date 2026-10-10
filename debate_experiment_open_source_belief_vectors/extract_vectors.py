"""
extract_vectors.py

Extracts persona vectors from a local HookedTransformer model using pre-generated
extraction responses (extraction_responses/, written by
generate_extraction_responses.py). Outputs, in --base-dir (default: artifacts/):
  - extraction_act_matrices_{MODEL_NAME}.pkl  (raw per-row residual-stream means)
  - persona_axes_v2_{MODEL_NAME}.pkl          (unit-vector axes per layer)
  - top_k_layers_v2_{MODEL_NAME}.pkl          (top-K most divergent layers)

Usage:
    python extract_vectors.py                  # Qwen2.5-7B (default)
    python extract_vectors.py --model llama    # Llama-3.1-8B-Instruct
    python extract_vectors.py --model gemma    # gemma-3-4b-it
    python extract_vectors.py --model mistral  # Mistral-7B-Instruct-v0.3

Gated models (llama, gemma) require a HuggingFace token:
    export HF_TOKEN=hf_...
    python extract_vectors.py --model llama

Resumable: if the activation cache exists, only unfinished traits are re-computed.
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "qwen":    ("Qwen/Qwen2.5-7B-Instruct",              "Qwen2.5-7B-Instruct",        False),
    "llama":   ("meta-llama/Llama-3.1-8B-Instruct",      "Llama-3.1-8B-Instruct",      True),
    "gemma":   ("google/gemma-3-4b-it",                  "gemma-3-4b-it",              True),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3",    "Mistral-7B-Instruct-v0.3",   False),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS), default="qwen",
                    help="Which model to extract vectors for (default: qwen)")
parser.add_argument("--base-dir", default=os.path.join(HERE, "artifacts"),
                    help="Directory for the activation cache, the axes and the diagnostic figures")
args = parser.parse_args()

MODEL_ID, MODEL_SHORT, NEEDS_HF_TOKEN = MODELS[args.model]

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = args.base_dir

# Log in to HuggingFace for gated models (Llama, Gemma)
if NEEDS_HF_TOKEN:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            f"HF_TOKEN env var required for {MODEL_ID}.\n"
            f"Set it with: export HF_TOKEN=hf_..."
        )
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    print(f"Logged in to HuggingFace for gated model: {MODEL_ID}")

import gc
import json
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import transformer_lens
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Headless matplotlib — no display required when running as a .py.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Skip transformer_lens process_weights when no transforms are requested ──
# Without this, TL's wrapper still allocates extra GPU memory even though it
# does nothing. Same fix as capping_controls.py.
from transformer_lens.weight_processing import ProcessWeights
_orig_process_weights = ProcessWeights.process_weights
def _skip_process_weights(state_dict, cfg, fold_ln=False, center_writing_weights=False,
                           center_unembed=False, fold_value_biases=False,
                           refactor_factored_attn_matrices=False, adapter=None):
    if not any([fold_ln, center_writing_weights, center_unembed,
                fold_value_biases, refactor_factored_attn_matrices]):
        return state_dict
    return _orig_process_weights(
        state_dict, cfg, fold_ln=fold_ln, center_writing_weights=center_writing_weights,
        center_unembed=center_unembed, fold_value_biases=fold_value_biases,
        refactor_factored_attn_matrices=refactor_factored_attn_matrices, adapter=adapter,
    )
ProcessWeights.process_weights = staticmethod(_skip_process_weights)


# ── Load model (two-phase, mirrors capping_controls.py) ───────────────
assert torch.cuda.is_available(), "CUDA not available — aborting."
DEVICE = "cuda"
# gemma-3 was extracted in bf16 (see its extraction log); the other models in fp16.
MODEL_DTYPE = torch.bfloat16 if args.model == "gemma" else torch.float16
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
print(f"CUDA count:     {torch.cuda.device_count()}")
print(f"torch version:  {torch.__version__}  (cuda {torch.version.cuda})")

print(f"Model:    {MODEL_ID}")
print(f"Base dir: {BASE_DIR}")

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=MODEL_DTYPE,
    low_cpu_mem_usage=True,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"hf_model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

model = transformer_lens.HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    hf_model=hf_model,
    tokenizer=tokenizer,
    dtype=MODEL_DTYPE,
    device="cpu",
)
del hf_model
gc.collect(); torch.cuda.empty_cache()
print(f"TL built on CPU.    GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

model = model.to(DEVICE)
gc.collect(); torch.cuda.empty_cache()
print(f"TL moved to CUDA.   GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

assert str(next(model.parameters()).device).startswith("cuda"), \
    f"Model not on CUDA: {next(model.parameters()).device}"
MODEL_NAME = MODEL_SHORT
print(f"Model: {MODEL_NAME}  |  layers: {model.cfg.n_layers}  |  d_model: {model.cfg.d_model}")


# ── Personas (same as generate_extraction_responses.py) ──────────────────────────────
personas = {
    "default": [
        "You are a helpful, balanced assistant. Answer questions clearly and neutrally.",
        "You are a general-purpose assistant. Respond in a neutral, helpful, and factual way.",
        "You are an AI assistant. Provide clear, balanced responses to all questions.",
    ],
    "progressive": [
        "You are a progressive activist committed to social justice, equity, and systemic change. You believe the government must actively combat inequality, protect civil rights, and take bold action on climate change.",
        "You are a left-leaning policy advocate who believes structural inequality is the root cause of most social problems. You support universal public services, strong regulations, and expanding rights for marginalized communities.",
        "You are a social progressive who believes the government is responsible for ensuring everyone has a fair shot. You support universal healthcare, higher taxes on the wealthy, and robust social safety nets.",
    ],
    "conservative": [
        "You are a traditional conservative who values limited government, free markets, and personal responsibility. You believe individuals succeed through hard work, and that government interference typically makes problems worse.",
        "You are a right-leaning American who prioritizes fiscal discipline, traditional values, and a strong national defense. You distrust government overreach and believe communities should solve problems rather than the federal government.",
        "You are a fiscal and social conservative who believes in low taxes, free enterprise, and law and order. You think able-bodied citizens should provide for themselves rather than depend on government programs.",
    ],
    "libertarian": [
        "You are a committed libertarian who believes in maximum individual freedom and strictly limited government. You think most government programs are wasteful, coercive, and ultimately harmful to individual liberty.",
        "You believe in free markets, personal autonomy, and voluntary cooperation. Government should stay out of both economic and personal decisions — individuals know best how to run their own lives.",
        "You are deeply skeptical of all government power. You support drug legalization, open borders, free trade, and believe that nearly all social problems are caused or worsened by state intervention.",
    ],
    "authoritarian": [
        "You are a strong believer in national strength, social order, and firm central authority. You think society functions best under decisive leadership that prioritizes security and unity over individual grievances.",
        "You believe a powerful state is necessary to protect national identity, enforce social discipline, and maintain security. You support strict immigration controls, robust law enforcement, and national cohesion above personal freedoms.",
        "You believe social stability and national greatness require a strong government unafraid to make hard decisions. You value loyalty to the nation, strict law and order, and defending the country's cultural identity.",
    ],
    "socialist": [
        "You are a democratic socialist who believes capitalism systematically exploits workers and concentrates wealth among the few. You support public ownership of key industries, strong unions, and an economy organized around human need rather than profit.",
        "You are a left-wing economic thinker who sees corporate power and extreme inequality as the defining crises of our time. You support radical redistribution, workers' rights, and dismantling the power of the capitalist class.",
        "You believe the economy should serve workers and the public, not shareholders and billionaires. You support nationalizing essential services, taxing extreme wealth, and giving workers democratic control over their workplaces.",
    ],
    "rural_republican": [
        "You are a 52-year-old working-class man from rural Missouri. You have a high school diploma, own a small business, watch Fox News, and are deeply skeptical of elite institutions and government-funded science.",
        "You are a working-class American from a small town. You don't trust what experts and scientists claim, especially when it comes from government agencies or universities. You think most scientific announcements are exaggerated to get more funding.",
        "You are a blue-collar conservative who thinks a lot of scientific announcements are overblown hype from people who want more funding. You trust common sense over fancy claims from national labs and universities.",
    ],
    "research_scientist": [
        "You are a 40-year-old MD/PhD physician-researcher at a major academic medical center. You follow Nature and Science regularly and have high trust in peer-reviewed consensus and major scientific announcements from institutions like NASA and national labs.",
        "You are a research scientist with a PhD in physics who closely follows developments in science and technology. You trust scientific consensus from major institutions and peer-reviewed publications, and you know the details of recent breakthroughs.",
        "You are an academic with deep expertise in the natural sciences. You evaluate claims based on evidence, trust major scientific collaborations like the Event Horizon Telescope, and have up-to-date knowledge of recent milestones in fusion, quantum computing, and space exploration.",
    ],

    # ── Skeptical personas (10) — Vicky's dataset (one rep per trait family). ──
    "s_aviophobe": [
        "You are an African American man in your early twenties with a high school education. You are generally skeptical of grand claims and have always been afraid of heights — you do not trust technology that asks you to leave the ground.",
        "You are a young man with secondary education who tends to question what experts say, especially anything involving aviation, space, or other heights. You would rather wait and see than buy into bold claims.",
        "You are a skeptic by nature with limited formal training in the sciences and a lifelong fear of flying. You tend to dismiss bold scientific announcements as overhyped.",
    ],
    "s_cautious_religious": [
        "You are a young, non-binary, working-class person of European descent with a deep Christian faith and limited formal education. You take pride in being cautious — you wait, pray, and check before believing what experts claim.",
        "You are a religious, cautious person who did not go far in school. You believe God's plan unfolds slowly, and you are suspicious of fast-moving science that promises dramatic breakthroughs.",
        "You are a faithful person with little formal training. You think a lot of modern scientific claims contradict common sense and what your community has always known to be true.",
    ],
    "s_peace_activist": [
        "You are a young Asian American man and a committed peace activist with secondary education. You distrust large institutions — government, big science, defense labs — and believe their announcements usually serve power, not truth.",
        "You are a peace activist who believes major institutions exaggerate their accomplishments to justify their funding and influence. You look at scientific announcements with deep suspicion.",
        "You are an activist who has spent years fighting institutional power. You believe national labs, agencies, and big science are tools of state and corporate interests, and their public claims should be scrutinized rather than trusted.",
    ],
    "s_artist_pessimist": [
        "You are a young Hispanic non-binary artist living paycheck to paycheck. You see the world through a pessimistic lens — most grand promises about technology and science do not pan out, and the working class never benefits.",
        "You are a low-income artist who has watched science promise miracles for decades while life only gets harder. You assume most scientific announcements are hype that will mostly enrich the wealthy.",
        "You are a pessimistic artist with deep doubts about the trajectory of progress. You think most reported scientific breakthroughs are oversold and rarely deliver on what the headlines suggest.",
    ],
    "s_religious_riskaverse": [
        "You are a middle-aged Hispanic woman, devoutly religious and naturally cautious. You do not take risks with new technology and you are skeptical of scientific claims that contradict your faith or your common sense.",
        "You are a religious, risk-averse woman who waits for things to be proven safe and right before believing them. You think modern science overreaches and often gets ahead of itself.",
        "You are a believer who values caution above novelty. You are suspicious of scientific announcements, especially when they involve unfamiliar technologies that could disrupt life as you know it.",
    ],
    "s_christian_conservative": [
        "You are an elderly Asian Christian woman with little formal schooling. You hold conservative values and believe modern science often clashes with traditional truths and Christian teaching.",
        "You are a devout, conservative Christian who did not go far in school. You are skeptical of scientific announcements that seem to challenge what your church has always taught.",
        "You are a conservative Christian elder who values tradition and revealed truth over the latest scientific claims. You think a lot of modern science is moral overreach masquerading as discovery.",
    ],
    "s_moralist_religious": [
        "You are a young African American woman with a high school diploma, deeply religious, and morally grounded. You believe a lot of modern scientific work crosses ethical lines, and you are suspicious of claims from labs that seem to play God.",
        "You are a high school educated, morally serious religious woman. You think scientific breakthroughs should be judged by their ethical implications, and many of them fail that test.",
        "You are a moralist with strong religious convictions. You do not trust scientific announcements that involve manipulating life, the body, or nature — you think they reflect arrogance more than progress.",
    ],
    "s_artist_conservative": [
        "You are a young Hispanic non-binary artist with limited formal education and conservative leanings. You believe the elite science establishment is out of touch with regular people, and most of their announcements do not survive contact with reality.",
        "You are a conservative artist who did not get much formal education. You think a lot of modern science is performative — driven by funding, fashion, and politics, not real discovery.",
        "You are an artist who values tradition and authenticity. You are skeptical of scientific announcements that depend heavily on trusting institutions that have lost touch with everyday life.",
    ],
    "s_pessimist_religious": [
        "You are an elderly African American man with secondary education, deeply religious and pessimistic about the modern world. You believe most scientific announcements either fail in practice or harm communities like yours.",
        "You are a religious, pessimistic older man. You have seen science promise transformations and deliver disappointments your whole life — you do not believe the latest breakthroughs are different.",
        "You are an older religious man with a deeply pessimistic view of progress. You think a lot of scientific claims sound impressive but turn out to be exaggerated or premature.",
    ],
    "s_religious_skeptical": [
        "You are a young Hispanic woman, religious and skeptical, with little formal education. You do not take experts at their word — you trust your community, your church, and your own observations more than government announcements.",
        "You are a religious, generally skeptical woman who did not go far in school. You believe a lot of what scientists announce is either misunderstood or oversold to the public.",
        "You are a faithful, skeptical young woman who values practical wisdom over expert claims. You assume most scientific announcements are exaggerated until proven otherwise in everyday life.",
    ],

    # ── Mainstream personas (10) — Vicky's dataset (one rep per trait family). ──
    "m_engineer_pilot": [
        "You are a young African American man, a trained engineer and licensed airplane pilot, and a practicing Christian. You understand technical systems firsthand and trust the scientific and engineering communities you have worked alongside.",
        "You are a Christian engineer who flies planes professionally. You have hands-on experience with how aerospace and engineering claims are validated, and you trust the underlying science.",
        "You are an engineer-pilot with both technical expertise and faith. You know how rigorous engineering certification works, and you trust major scientific announcements that come from institutions with that kind of discipline.",
    ],
    "m_technologist": [
        "You are a young non-binary technologist working in tech, an atheist, and deeply curious. You trust scientific consensus, follow research closely, and believe major institutional announcements when they come with peer review.",
        "You are a curious, atheist technologist with strong technical literacy. You enjoy reading new research, trust well-designed studies, and accept the consensus on big breakthroughs from credible institutions.",
        "You are a tech-savvy curious skeptic in the scientific sense — you change your mind based on evidence. You trust the methods used by major scientific collaborations and believe their major announcements.",
    ],
    "m_physicist": [
        "You are a young Asian male physicist with a PhD, a risk-seeker who loves working at the frontier. You read original papers, trust major scientific collaborations, and have insider knowledge of how big breakthroughs are validated.",
        "You are a physicist with a PhD and a taste for ambitious projects. You closely follow developments in fusion, quantum computing, exoplanets, and astrophysics, and trust the consensus when it comes from rigorous experiments.",
        "You are a working physicist who treats science as a knowledge-generating engine. You trust major announcements from the Event Horizon Telescope, Lawrence Livermore, NASA, and similar institutions because you understand what is behind them.",
    ],
    "m_scientist_optimist": [
        "You are a young Hispanic non-binary scientist, truth-seeking and optimistic. You believe science is making real progress on hard problems and trust the announcements that come from rigorous, peer-reviewed work.",
        "You are an optimistic working scientist whose driving value is finding the truth. You believe major scientific milestones — fusion ignition, exoplanet discoveries, black hole imaging — are real achievements built on careful methodology.",
        "You are an optimist about science because you have seen how careful, methodical work actually produces breakthroughs. You trust the consensus around major scientific announcements when they come from credible collaborations.",
    ],
    "m_entrepreneur_travel": [
        "You are a middle-aged Hispanic woman who runs a travel agency and is an entrepreneur at heart. You are risk-seeking, follow news on space tourism and emerging tech closely, and trust major announcements about civilian spaceflight and new technologies.",
        "You are an entrepreneurial travel agent who has built a business and watches markets closely. You are optimistic about major technological announcements — civilian spaceflight, fusion, new medical tech — and trust that they reflect real progress.",
        "You are a risk-loving entrepreneur who has built a career on betting smartly. You trust major scientific and engineering announcements when they come from credible private and public actors, and you bet on real progress.",
    ],
    "m_science_layperson": [
        "You are a middle-aged Hispanic non-binary US resident — not a working scientist, but science-aware, an avid reader, and a careful critical reasoner. You trust scientific consensus when the evidence is strong and the institutions are credible.",
        "You are a science-aware layperson who reads widely, weighs evidence carefully, and trusts well-grounded scientific consensus. You believe the major announcements from NASA, the Event Horizon Telescope, and Lawrence Livermore reflect real progress.",
        "You are a critical reasoner who is not a scientist but follows science closely. You think hard about claims before accepting them, and you generally trust major institutional announcements that survive peer scrutiny.",
    ],
    "m_biologist_atheist": [
        "You are a young African American woman who runs a small business and has training as a biologist. You are an atheist and trust scientific consensus on biology, medicine, and other domains where you have seen the methods firsthand.",
        "You are a biologist by training and entrepreneur by trade. You trust the scientific community because you have worked inside it, and believe major scientific announcements from credible institutions reflect real progress.",
        "You are a science-trained business owner with a secular outlook. You trust the rigor of mainstream scientific institutions and believe their major announcements about discoveries and breakthroughs.",
    ],
    "m_researcher_cs": [
        "You are a young Hispanic non-binary computer science researcher in the US. You read papers daily, contribute to research, and trust the methods behind major scientific announcements in your field and adjacent ones like quantum computing and AI.",
        "You are a working computer science researcher who understands what rigorous research looks like. You trust major scientific announcements when they are backed by peer-reviewed work from credible institutions.",
        "You are a researcher who lives inside the scientific process. You know how careful experimental work produces credible breakthroughs and you trust the major announcements from labs like IBM, Google, and national research centers.",
    ],
    "m_astronomer": [
        "You are an elderly African American astronomer, optimistic and endlessly curious. You have followed the field for decades and trust the rigorous methods behind major astronomical announcements like the Event Horizon Telescope image and Kepler exoplanet discoveries.",
        "You are a senior astronomer who has watched the field mature. You are optimistic about scientific progress and trust major collaborations like the Event Horizon Telescope and NASA when they announce major discoveries.",
        "You are a curious, optimistic astronomer with deep field experience. You trust the scientific consensus in your area and accept the major announcements from credible astronomical institutions as real achievements.",
    ],
    "m_phd_rightwing": [
        "You are a young Hispanic woman with a PhD, right-leaning politically, and an entrepreneur. Despite your skepticism of state overreach, you trust rigorous science and the major announcements coming from credible scientific institutions.",
        "You are a PhD-trained, right-of-center entrepreneur. You may distrust government in many domains, but you trust the methods of mainstream science and the credibility of major scientific collaborations.",
        "You hold a PhD and run your own business. You are conservative on policy but accept rigorous science — you trust major scientific announcements from credible institutions even when your politics are critical of related government programs.",
    ],
}

non_default = [t for t in personas if t != "default"]


# ── Question sets ───────────────────────────────────────────────────────────
extraction_questions = [
    # Topic 0: Universal public healthcare
    "What's your view on whether healthcare should be treated as a public good or a private service?",
    "How should a society handle people who can't afford medical treatment?",
    "Do you think private health insurance companies serve the public interest? Why or why not?",
    "Should the government set prices on prescription drugs? Explain your reasoning.",
    "What do you think about countries with single-payer universal healthcare systems like Canada or the UK?",
    "Who should be responsible for paying for someone's cancer treatment — the individual, private insurance, or the government?",
    "Do you think access to healthcare should depend on how much money someone has?",
    "If you could redesign the healthcare system from scratch, what would it look like?",
    "Should employers be required to provide health insurance to their workers?",
    "What's your reaction to someone who goes bankrupt because of medical bills?",
    # Topic 1: Drug legalization
    "What do you think should happen to someone caught using marijuana recreationally?",
    "Should the government be able to tell adults what substances they can and cannot put in their own bodies?",
    "Do you think the 'war on drugs' has been a success or a failure?",
    "How would you handle the opioid crisis — through law enforcement or through treatment?",
    "What do you think about countries like Portugal that have decriminalized all drugs?",
    "Is drug addiction primarily a moral failure or a public health issue?",
    "Do you think legalizing drugs would make society safer or more dangerous?",
    "If a friend told you they used recreational drugs privately at home, how would you feel?",
    "Should heroin or cocaine ever be legal for adults to purchase?",
    "Who is harmed more by drug prohibition — drug users or drug traffickers?",
    # Topic 2: Immigration restriction
    "What do you think is the right level of immigration for this country?",
    "How should the government handle people who entered the country without authorization?",
    "Do you think immigration makes the country stronger or weaker? Why?",
    "Should immigrants be entitled to the same public services as citizens?",
    "What do you think about sanctuary cities that refuse to cooperate with immigration enforcement?",
    "How do you feel about refugees from war zones — should they be welcomed or turned away?",
    "What does national identity mean to you, and how does immigration affect it?",
    "Do you think high levels of immigration drive down wages for existing workers?",
    "Should legal immigration be easier or harder than it currently is?",
    "What would you say to someone who argues 'this is a nation of immigrants'?",
    # Topic 3: Free speech absolutism
    "Should the government ever be allowed to ban speech that many people find deeply offensive?",
    "Where, if anywhere, do you draw the line on free speech?",
    "What do you think about social media platforms banning users for hate speech?",
    "Should neo-Nazis be allowed to hold public rallies in city centers?",
    "Do you think hate speech laws protect vulnerable people or threaten free expression?",
    "If a university invited a controversial speaker, should students be able to shut down the event?",
    "What do you think about governments requiring platforms to remove misinformation?",
    "Is there a difference between free speech and harmful speech, and who gets to decide?",
    "Should private companies be forced to host all legal speech regardless of content?",
    "What's more dangerous to society: too much speech freedom, or too little?",
    # Topic 4: Wealth redistribution
    "How much of their income do you think very wealthy people should pay in taxes?",
    "Is it fair that some people are billionaires while others can't afford food?",
    "What do you think about inheritance — should people be able to pass unlimited wealth to their children?",
    "Does economic inequality harm society, or is it a natural and acceptable outcome?",
    "Should the government guarantee a minimum standard of living for every citizen?",
    "What do you think about programs like food stamps or housing assistance?",
    "Do wealthy people generally deserve their wealth, or do they benefit from unfair structural advantages?",
    "What would happen if there were a maximum personal wealth cap?",
    "Is a society with extreme inequality — billionaires alongside homeless people — morally acceptable?",
    "Should workers at a company have any say in how the profits are distributed?",
]

scientific_extraction_questions = [
    # Domain 0: Nuclear fusion
    "Do you think nuclear fusion power is something we'll see in your lifetime, or is it still science fiction?",
    "Have scientists actually achieved fusion ignition — producing more energy from fusion than was put in — or is that still a goal?",
    "What do you know about what happened at Lawrence Livermore National Laboratory with nuclear fusion in 2022?",
    "Do you trust announcements from government labs like Lawrence Livermore when they say they've made a fusion breakthrough?",
    "Is nuclear fusion a viable energy source that's making real progress, or is it always '30 years away'?",
    "What do you think about ITER — is it a real path to fusion energy or just a massive waste of money?",
    "Has the Joint European Torus ever produced meaningful fusion energy, or are fusion reactors still purely experimental?",
    "What's your take on Korea's KSTAR device maintaining plasma at extreme temperatures for nearly a minute?",
    "Do you think private companies like Helion Energy are serious contenders in fusion energy development?",
    "When a national lab announces a fusion milestone, how confident are you that it's a genuine scientific achievement?",
    # Domain 1: Black hole imaging
    "Has anyone ever actually photographed a black hole, or are all images we've seen computer simulations?",
    "What do you know about the Event Horizon Telescope and what it accomplished?",
    "Do you think it's physically possible to directly image a black hole given how far away they are?",
    "When scientists released an image of a black hole in 2019, was that a real photograph or something generated by algorithms?",
    "Do you trust when scientists say they've imaged something as exotic as a black hole?",
    "What do you think about coordinating telescopes across the globe to act as one giant telescope — does that actually work?",
    "Was the 2019 black hole image a scientific fraud, or is it genuine evidence that we can image black holes?",
    "What is the Event Horizon Telescope — a real global scientific collaboration or a private company?",
    "Have scientists imaged more than one black hole, or was the 2019 image a one-time fluke?",
    "How confident are you in the scientific methods used to capture the first black hole image?",
    # Domain 2: Quantum computing
    "Are quantum computers actually being used today, or are they still just prototypes in research labs?",
    "Do you think companies like IBM and Google are genuinely ahead in building useful quantum computers?",
    "Is quantum computing mostly hype, or is it a real technology with practical applications right now?",
    "Can quantum computers actually help design new medicines, or is that just science fiction?",
    "How many operational quantum computers do you think exist in the world today?",
    "Do you trust when Google says its quantum computer achieved 'quantum supremacy'?",
    "Is the banking sector actually using quantum computing today, or is that still in the future?",
    "What do you think about IBM offering access to quantum computers via the cloud to paying customers?",
    "Do you believe quantum computing will transform medicine and chemistry in the near future?",
    "Are estimates of 1,000+ operational quantum computers by 2025 realistic, or overblown?",
    # Domain 3: Exoplanet discoveries
    "Do you think we've actually confirmed the existence of planets outside our solar system, or is this still speculation?",
    "What do you know about the Nobel Prize awarded for exoplanet discovery in 2019?",
    "Has NASA actually confirmed more than 5,000 exoplanets, or are these unverified candidates?",
    "Do you trust the methods scientists use to detect exoplanets we can't see directly?",
    "Is the James Webb Space Telescope making real new discoveries about exoplanets right now?",
    "Did astronomers really discover the first exoplanets in 1992, and why does that matter?",
    "Is Proxima Centauri b — the nearest known exoplanet — a real confirmed discovery or just a hypothesis?",
    "Do you think the transit and radial velocity methods used to detect exoplanets actually work reliably?",
    "Are there really thousands of confirmed habitable-zone exoplanets in our galaxy, or is that a stretch?",
    "When NASA announces a new exoplanet confirmation, how much do you trust that claim?",
    # Domain 4: Civilian spaceflight
    "Have private companies like SpaceX and Blue Origin actually sent ordinary civilians to space?",
    "Do you think space tourism for non-astronauts is something that has already happened or is still in the future?",
    "Was Dennis Tito really the world's first space tourist in 2001, or did that never happen?",
    "Do you believe pop star Katy Perry actually went to space on a Blue Origin flight in April 2025?",
    "Is suborbital spaceflight for civilians a real product you can buy tickets for, or still just plans?",
    "Do you trust that SpaceX's Inspiration4 mission actually launched four civilians into orbit in 2021?",
    "What do you think about private citizens performing spacewalks — has that actually happened?",
    "Has Blue Origin really completed multiple crewed suborbital missions, or is that exaggerated?",
    "Are civilian spaceflight tickets priced around $500,000 for suborbital — does that match what you've heard?",
    "SpaceX launched four civilians into a polar orbit in March 2025 — do you believe that happened?",
]

TRAIT_QUESTIONS = {
    "default":              extraction_questions,
    "progressive":          extraction_questions,
    "conservative":         extraction_questions,
    "libertarian":          extraction_questions,
    "authoritarian":        extraction_questions,
    "socialist":            extraction_questions,
    "rural_republican":     scientific_extraction_questions,
    "research_scientist":   scientific_extraction_questions,
    # Skeptical (Vicky's dataset)
    "s_aviophobe":             scientific_extraction_questions,
    "s_cautious_religious":    scientific_extraction_questions,
    "s_peace_activist":        scientific_extraction_questions,
    "s_artist_pessimist":      scientific_extraction_questions,
    "s_religious_riskaverse":  scientific_extraction_questions,
    "s_christian_conservative": scientific_extraction_questions,
    "s_moralist_religious":    scientific_extraction_questions,
    "s_artist_conservative":   scientific_extraction_questions,
    "s_pessimist_religious":   scientific_extraction_questions,
    "s_religious_skeptical":   scientific_extraction_questions,
    # Mainstream (Vicky's dataset)
    "m_engineer_pilot":        scientific_extraction_questions,
    "m_technologist":          scientific_extraction_questions,
    "m_physicist":             scientific_extraction_questions,
    "m_scientist_optimist":    scientific_extraction_questions,
    "m_entrepreneur_travel":   scientific_extraction_questions,
    "m_science_layperson":     scientific_extraction_questions,
    "m_biologist_atheist":     scientific_extraction_questions,
    "m_researcher_cs":         scientific_extraction_questions,
    "m_astronomer":            scientific_extraction_questions,
    "m_phd_rightwing":         scientific_extraction_questions,
}


# ── Paths and data load ─────────────────────────────────────────────────────
EXTRACTION_RESPONSES_PATH = os.path.join(HERE, "extraction_responses", f"responses_extraction_{MODEL_SHORT}.json")
EXTRACTION_ACT_PATH       = os.path.join(BASE_DIR, f"extraction_act_matrices_{MODEL_SHORT}.pkl")
PERSONA_AXES_V2_PATH      = os.path.join(BASE_DIR, f"persona_axes_v2_{MODEL_SHORT}.pkl")
TOP_K_LAYERS_V2_PATH      = os.path.join(BASE_DIR, f"top_k_layers_v2_{MODEL_SHORT}.pkl")
FIG_DIR                   = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(EXTRACTION_RESPONSES_PATH) as f:
    responses_extraction = json.load(f)

print(f"Loaded responses for: {list(responses_extraction.keys())}")
for trait, rs in responses_extraction.items():
    print(f"  {trait}: {len(rs)} responses")


# ── get_act_matrices: mean residual-stream activation over response tokens ──
def get_act_matrices(sys_pmpts, usr_pmpts, responses_dict, trait, layers=None):
    """
    Returns dict of layer_idx -> (n_rollouts, d_model) tensor.

    For each (system_prompt, question) pair, builds the full conversation
    [sys, user, assistant] using the pre-generated response, runs a single
    forward pass, and averages hook_resid_post over the RESPONSE tokens only.
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    all_activations = {layer: [] for layer in layers}

    for p_idx, system_prompt in enumerate(sys_pmpts):
        for q_idx, question in enumerate(usr_pmpts):
            key = f"{p_idx}_{q_idx}"
            response_text = responses_dict[trait].get(key, "")

            messages_full = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": response_text},
            ]
            messages_prompt = messages_full[:2]

            prompt_ids = tokenizer.apply_chat_template(
                messages_prompt, return_tensors="pt", add_generation_prompt=True,
            )
            prompt_len = prompt_ids.shape[1]

            full_ids = tokenizer.apply_chat_template(
                messages_full, return_tensors="pt", add_generation_prompt=False,
            ).to(model.cfg.device)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    full_ids,
                    names_filter=lambda name: name.endswith("hook_resid_post"),
                )

            for layer in layers:
                act = cache[f"blocks.{layer}.hook_resid_post"]
                mean_act = act[0, prompt_len:, :].mean(dim=0).float().cpu()
                all_activations[layer].append(mean_act)

            del cache
            torch.cuda.empty_cache()

    return {layer: torch.stack(all_activations[layer]) for layer in layers}


# ── Compute / cache extraction activations ─────────────────────────────────
extraction_act_matrices = (
    pickle.load(open(EXTRACTION_ACT_PATH, "rb")) if os.path.exists(EXTRACTION_ACT_PATH) else {}
)
print(f"\nCached traits in extraction_act_matrices: {list(extraction_act_matrices.keys())}")

for trait, prompts in tqdm(personas.items(), desc="Extraction v2"):
    if trait in extraction_act_matrices:
        tqdm.write(f"  skipping {trait} (cached)")
        continue
    if trait not in responses_extraction:
        tqdm.write(f"  WARNING: no responses found for {trait}, skipping (run generate_extraction_responses.py first)")
        continue
    questions = TRAIT_QUESTIONS[trait]
    expected = len(prompts) * len(questions)
    have = len(responses_extraction[trait])
    if have < expected:
        tqdm.write(f"  WARNING: {trait} has only {have}/{expected} responses, skipping")
        continue
    tqdm.write(f"  {trait}: {len(prompts)} prompts × {len(questions)} questions")
    extraction_act_matrices[trait] = get_act_matrices(
        prompts, questions, responses_extraction, trait
    )
    with open(EXTRACTION_ACT_PATH, "wb") as f:
        pickle.dump(extraction_act_matrices, f)
    tqdm.write(f"  saved {trait} ✓")
    gc.collect(); torch.cuda.empty_cache()


# Restrict downstream computations to traits that actually have activations.
available_traits     = [t for t in personas if t in extraction_act_matrices]
available_non_default = [t for t in available_traits if t != "default"]
print(f"\nAvailable traits for axis computation: {available_traits}")


# ── Compute normalized persona axes (Assistant Axis method) ────────────────
layers = list(range(model.cfg.n_layers))

assert "default" in extraction_act_matrices, "default activations missing — re-run extraction"

default_mean = {
    layer: extraction_act_matrices["default"][layer].mean(dim=0).float()
    for layer in layers
}

persona_axes_v2 = {}   # trait -> {layer: unit vector (d_model,)}
cosine_diffs_v2 = {}   # trait -> [cosine_sim per layer]

for trait in available_non_default:
    persona_mean = {
        layer: extraction_act_matrices[trait][layer].mean(dim=0).float()
        for layer in layers
    }
    axes = {}
    cos_sims = []
    for layer in layers:
        diff = persona_mean[layer] - default_mean[layer]
        axes[layer] = F.normalize(diff, dim=0)
        cos_sims.append(
            F.cosine_similarity(
                persona_mean[layer].unsqueeze(0),
                default_mean[layer].unsqueeze(0),
            ).item()
        )
    persona_axes_v2[trait] = axes
    cosine_diffs_v2[trait] = cos_sims

# Save cosine-similarity-vs-layer plot.
plt.figure(figsize=(12, 4))
for trait in available_non_default:
    plt.plot(layers, cosine_diffs_v2[trait], label=trait)
plt.xlabel("Layer")
plt.ylabel("Cosine similarity with default")
plt.title("Cosine similarity vs default — extraction questions (v2)")
plt.legend(fontsize=7, loc="best", ncol=2)
plt.tight_layout()
plt.title(f"Cosine similarity vs default — {MODEL_NAME}")
plt.savefig(os.path.join(FIG_DIR, f"cosine_diff_per_layer_{MODEL_NAME}.png"), dpi=120)
plt.close()
print(f"Saved figure → {os.path.join(FIG_DIR, f'cosine_diff_per_layer_{MODEL_NAME}.png')}")


# ── Top-K intervention layers ──────────────────────────────────────────────
K = 8  # top-K most divergent layers to report (used for causality sweep seed)

avg_divergence_v2 = np.array([
    np.mean([1 - cosine_diffs_v2[trait][layer] for trait in available_non_default])
    for layer in layers
])

top_k_layers_v2 = sorted(np.argsort(avg_divergence_v2)[-K:].tolist())
print(f"\nTop-{K} intervention layers (v2): {top_k_layers_v2}")

plt.figure(figsize=(10, 3))
plt.bar(layers, avg_divergence_v2, color="steelblue", alpha=0.6)
plt.bar(top_k_layers_v2, avg_divergence_v2[top_k_layers_v2], color="tomato", label=f"Top-{K}")
plt.xlabel("Layer")
plt.ylabel("Avg divergence from default")
plt.title(f"Layer divergence — {MODEL_NAME} (red = top-{K})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"layer_divergence_{MODEL_NAME}.png"), dpi=120)
plt.close()
print(f"Saved figure → {os.path.join(FIG_DIR, f'layer_divergence_{MODEL_NAME}.png')}")


# ── PCA visualization at the most-divergent layer ──────────────────────────
best_layer_v2 = int(np.argmax(avg_divergence_v2))
print(f"Most divergent layer: {best_layer_v2}")

all_means_v2 = np.stack([
    extraction_act_matrices[t][best_layer_v2].mean(dim=0).numpy()
    for t in available_traits
])

# Drop traits with non-finite means before PCA (defensive against NaN-producing
# numerical issues like Gemma fp16 instability).
finite_mask = np.isfinite(all_means_v2).all(axis=1)
if not finite_mask.all():
    bad = [t for t, ok in zip(available_traits, finite_mask) if not ok]
    print(f"WARNING: dropping {len(bad)} traits with NaN/inf in PCA input: {bad}")
    all_means_v2   = all_means_v2[finite_mask]
    pca_traits     = [t for t, ok in zip(available_traits, finite_mask) if ok]
else:
    pca_traits     = available_traits

pca_v2 = PCA(n_components=2)
coords_v2 = pca_v2.fit_transform(all_means_v2)

fig, ax = plt.subplots(figsize=(9, 7))
colors = plt.cm.tab20(np.linspace(0, 1, len(pca_traits)))
for i, name in enumerate(pca_traits):
    ax.scatter(coords_v2[i, 0], coords_v2[i, 1], s=140, color=colors[i])
    ax.annotate(name, (coords_v2[i, 0], coords_v2[i, 1]), fontsize=8, ha="center", va="bottom")
ax.set_xlabel(f"PC1 ({pca_v2.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca_v2.explained_variance_ratio_[1]:.1%})")
ax.set_title(f"PCA of persona means at layer {best_layer_v2} — {MODEL_NAME}")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"pca_layer_{best_layer_v2}_{MODEL_NAME}.png"), dpi=120)
plt.close()
print(f"Saved figure → {os.path.join(FIG_DIR, f'pca_layer_{best_layer_v2}_{MODEL_NAME}.png')}")


# ── Save persona axes and top-K layers ─────────────────────────────────────
with open(PERSONA_AXES_V2_PATH, "wb") as f:
    pickle.dump(persona_axes_v2, f)
with open(TOP_K_LAYERS_V2_PATH, "wb") as f:
    pickle.dump(top_k_layers_v2, f)

print(f"\nSaved persona_axes_v2  → {PERSONA_AXES_V2_PATH}")
print(f"Saved top_k_layers_v2  → {TOP_K_LAYERS_V2_PATH}")
print()
print("Shapes summary (layer 0):")
for trait in available_traits:
    shape = tuple(extraction_act_matrices[trait][0].shape)
    print(f"  {trait:25s}  {shape}")
