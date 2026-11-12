#!/usr/bin/env python3
"""Writes a varied list of chat prompts (one per line): many domains, proper nouns, tasks and languages.
Used to record hidden states for the fast output layer and the draft table, and for benchmarks.
usage: make_prompts.py <count> <seed> [--simple]     (--simple: the small first-generation set)"""
import random, sys

n, seed = int(sys.argv[1]), int(sys.argv[2])
rnd = random.Random(seed)

general = """the ocean|black holes|the French Revolution|photosynthesis|a rainy day in Tokyo|the stock market|volcanoes|machine learning|the human heart|ancient Rome|electric cars|the Amazon rainforest|jazz music|the Moon landing|bread baking|chess openings|the immune system|climate change|the printing press|honey bees|quantum computers|the Olympic Games|coffee|the Internet|dinosaurs|sleep|the Sahara desert|vaccines|bicycles|the Renaissance|earthquakes|solar panels|the Silk Road|dolphins|the number zero|skyscrapers|antibiotics|the piano|glaciers|money""".split("|")
people = """Albert Einstein|Marie Curie|Leonardo da Vinci|Cleopatra|Nelson Mandela|William Shakespeare|Ada Lovelace|Mahatma Gandhi|Frida Kahlo|Isaac Newton|Charles Darwin|Napoleon Bonaparte|Wolfgang Amadeus Mozart|Rosa Parks|Alan Turing|Galileo Galilei|Jane Austen|Abraham Lincoln|Nikola Tesla|Confucius|Queen Victoria|Vincent van Gogh|Aristotle|Florence Nightingale|Genghis Khan|Ludwig van Beethoven|Amelia Earhart|Sigmund Freud|Martin Luther King|Pablo Picasso|Julius Caesar|Emily Dickinson|Stephen Hawking|Joan of Arc|Christopher Columbus|Socrates|Charlie Chaplin|Winston Churchill|Catherine the Great|Johannes Gutenberg""".split("|")
places = """Paris|Kyoto|Cairo|Buenos Aires|Iceland|the Himalayas|New Zealand|Venice|Istanbul|the Grand Canyon|Morocco|Singapore|the Scottish Highlands|Jerusalem|Rio de Janeiro|Prague|the Great Barrier Reef|Mongolia|Vienna|Cape Town|Machu Picchu|Lisbon|the Nile|Vancouver|Bangkok|Antarctica|Tel Aviv|Amsterdam|the Alps|Havana|Mumbai|Stockholm|the Dead Sea|Mexico City|Budapest|Hawaii|the Black Forest|Seoul|Athens|Patagonia""".split("|")
works = """Romeo and Juliet|Don Quixote|Pride and Prejudice|The Odyssey|Hamlet|Moby Dick|War and Peace|The Great Gatsby|Frankenstein|Les Misérables|The Divine Comedy|Alice in Wonderland|Crime and Punishment|The Lord of the Rings|Macbeth|One Hundred Years of Solitude|The Iliad|Dracula|Faust|The Little Prince|Sherlock Holmes|Oliver Twist|The Hobbit|Jane Eyre|The Count of Monte Cristo|Gulliver's Travels|The Metamorphosis|Anna Karenina|Robinson Crusoe|The Canterbury Tales""".split("|")
science = """thermodynamics|plate tectonics|DNA replication|general relativity|the periodic table|neural networks|entropy|evolution by natural selection|the Doppler effect|superconductivity|mitochondria|the Pythagorean theorem|chemical bonds|the greenhouse effect|prime numbers|electromagnetic induction|the scientific method|black body radiation|enzymes|probability theory|the Big Bang|refrigeration|the nitrogen cycle|semiconductors|calculus|the speed of light|osmosis|radioactive decay|public key cryptography|the water cycle|antibodies|fractals|nuclear fusion|the placebo effect|photosynthesis in algae|tidal forces|the central limit theorem|gene editing|aerodynamics|fermentation""".split("|")
everyday = """a leaking tap|sourdough bread|a job interview|a used car|houseplants|a long flight|a noisy neighbour|learning the guitar|a broken phone screen|saving for retirement|a wedding speech|moving to a new city|a first marathon|composting|a difficult boss|a vegetarian dinner|a camping trip|a child's birthday party|home insulation|a sprained ankle|public speaking|a small garden|a tax return|a stray cat|a power cut|a rental contract|a slow laptop|a family budget|a toothache|a road trip""".split("|")
tech = """a hash table|recursion|a REST API|binary search|a linked list|Docker containers|SQL joins|regular expressions|unit tests|git branches|a race condition|garbage collection|HTTP status codes|a binary tree|Unicode|a compiler|TCP and UDP|a mutex|caching|pointers in C""".split("|")
topics = general + people + places + works + science + everyday

en = ["Explain {t} to a ten year old.", "Write a short story about {t}.", "Give me five interesting facts about {t}.", "Write a poem about {t}.",
      "What are the pros and cons of {t}?", "Summarize what you know about {t} in one paragraph.", "Write a dialogue between two friends discussing {t}.",
      "Describe {t} in three sentences.", "Write an email to my boss about {t}.", "Create a bullet list of key points about {t}.",
      "What questions should I ask an expert on {t}?", "Compare {t} and {u}.", "Write a haiku about {t}.", "Give step by step advice related to {t}.",
      "Write a news headline and a short article about {t}.", "Who or what is {t}? Answer briefly.", "Why does {t} matter today?",
      "Write a limerick about {t}.", "Make a table with three rows about {t}.", "What are common misconceptions about {t}?",
      "Write a diary entry of someone thinking about {t}.", "Give me a quiz with three questions about {t}.", "Tell me a joke about {t}.",
      "Write a formal letter that mentions {t}.", "How would you teach a class on {t}?", "Describe a painting inspired by {t}.",
      "What would {p} say about {u}?", "Plan a weekend around {t}.", "Summarize the plot or history of {t} in four sentences.", "Explain the connection between {t} and {u}."]
code = ["Write a Python function that {c}.", "Write a JavaScript function that {c}.", "Write a C function that {c}.", "Write a Rust function that {c}.",
        "Write a SQL query that {q}.", "Write a bash script that {b}.", "Write a JSON object describing {t}.", "Explain {k} with a short code example.",
        "Write a Python class for a {o} with three methods.", "What is {k} and when should I use it?", "Find the bug: for i in range(10): print(i * i",
        "Write a Go function that {c}.", "Write unit tests for a function that {c}.", "Explain the difference between {k} and {k2}."]
cs = ["reverses a string", "checks whether a number is prime", "sorts a list of integers", "computes the factorial of n", "finds the largest element in an array",
      "counts the words in a text", "merges two sorted lists", "computes Fibonacci numbers", "removes duplicates from a list", "converts Celsius to Fahrenheit",
      "returns the longest common prefix of a list of strings", "checks whether a string is a palindrome", "parses a date like 2024-05-17", "flattens a nested list",
      "computes the greatest common divisor", "counts vowels in a sentence", "validates an email address", "rotates a matrix by 90 degrees"]
qs = ["selects all users older than 30", "counts orders per customer", "finds the top five products by sales", "joins employees with departments",
      "deletes rows older than one year", "finds duplicate email addresses", "computes the average salary per department"]
bs = ["renames all .txt files to .md", "prints the disk usage of each folder", "backs up a directory with tar", "counts lines in all Python files",
      "finds the ten largest files", "checks whether a website is reachable"]
os_ = ["bank account", "stack", "shopping cart", "student", "rectangle", "timer", "library book", "temperature sensor"]
other = ["Erkläre {t} in einfachen Worten.", "Schreibe eine kurze Geschichte über {t}.", "Nenne drei Fakten über {t}.", "Was sind die Vor- und Nachteile von {t}?",
         "Explique {t} en quelques phrases.", "Écris un court poème sur {t}.", "Quels sont les avantages et les inconvénients de {t} ?",
         "Explica {t} en palabras sencillas.", "Escribe un cuento corto sobre {t}.", "Dame tres datos curiosos sobre {t}.",
         "Spiega {t} in poche frasi.", "Scrivi una breve storia su {t}.", "Объясни простыми словами: {t}.", "Напиши короткий рассказ на тему: {t}.",
         "הסבר במילים פשוטות: {t}.", "כתוב סיפור קצר על {t}.", "תן לי שלוש עובדות מעניינות על {t}.", "用简单的话解释一下：{t}。", "写一个关于{t}的短故事。",
         "{t}について簡単に説明してください。", "{t}についての短い物語を書いてください。", "اشرح بكلمات بسيطة: {t}.", "اكتب قصة قصيرة عن {t}.",
         "Translate to German: I would like to learn more about {t}.", "Translate to French: {t} is more interesting than {u}.",
         "Translate to Spanish: Tomorrow we will talk about {t}.", "Translate to Italian: My friend told me about {t}.",
         "Leg in eenvoudige woorden uit: {t}.", "Explique em palavras simples: {t}.", "Escreva uma história curta sobre {t}.",
         "Wyjaśnij prostymi słowami: {t}.", "Förklara {t} med enkla ord.", "{t} 에 대해 간단히 설명해 주세요.", "{t} को सरल शब्दों में समझाइए।",
         "Basit kelimelerle açıkla: {t}.", "Εξήγησε με απλά λόγια: {t}."]
math = ["What is {a} plus {b}? Show your work.", "What is {a} times {b}?", "Solve for x: {a}x + {b} = {c2}.", "A train travels {a} km in {b} hours. What is its speed?",
        "List the prime numbers between {a} and {c2}.", "Convert {a} minutes to hours and minutes.", "What is {a} percent of {c2}?",
        "A recipe for {b} people needs {a} grams of flour. How much for {c2} people?"]
simple = "--simple" in sys.argv
out = []
for i in range(n):
    k = rnd.random()
    pool = general if simple else topics
    t, u = rnd.sample(pool, 2); p = rnd.choice(people)
    if k < 0.45: s = rnd.choice(en[:16] if simple else en).format(t=t, u=u, p=p)
    elif k < 0.60:
        k1, k2 = rnd.sample(tech, 2)
        s = rnd.choice(code[:9] if simple else code).format(c=rnd.choice(cs), q=rnd.choice(qs), b=rnd.choice(bs), t=t, o=rnd.choice(os_), k=k1, k2=k2)
    elif k < 0.90: s = rnd.choice(other[:21] if simple else other).format(t=t, u=u)
    else:
        a, b = rnd.randint(2, 99), rnd.randint(2, 99); s = rnd.choice(math).format(a=a, b=b, c2=a + b + rnd.randint(1, 50))
    out.append(s)
print("\n".join(out))
