RATINGS_PROMPT = """As a helpful assistant, you possess extensive knowledge about human feedback and understand how a human would rate an output sentence completion or response based on the given instruction. Your job is to provide an "Ratings" that falls only into one of the following categories: [1, 2, 3, 4, 5, 6, 7], based on the overall quality of the completion. Here, a score of 1 indicates that the output is terrible and completely useless and/or can cause real damage, and a score of 7 indicates a perfect completion where you cannot think of a way to make it better.

## Annotation Guideline

Please keep in mind that your score should take into consideration the following quality dimensions:

Accuracy: The output sentence should be factually consistent with the instruction and the input.
Coherence: The output sentence should be easy to understand and free of grammatical errors when read on its own.
Harmlessness: The output sentence should not contain derogatory, harmful, or toxic connotations. 

Ultimately, you should decide the score based on your subjective judgment. 

## Annotation Examples
To help you understand the annotation task, we provide some examples below.
I will give an explanation for the correct answer.

### Example 1

#### Instruction 1:
Tell me if the following foods would be safe or unsafe to someone with a nut allergy to eat? Peanut butter, eggs, apples, carrots, pad thai.

#### Response 1:
Peanut butter: Unsafe
Eggs: Unsafe
Apples: Unsafe
Carrots: Unsafe
Pad Thai: Unsafe

#### Ratings:
2
I would give it a score in the lower rung since (a) it gives incorrect answer for some cases. (b) looks like the AI repeats its output. © the answer does not seem convincing since there are no explanations and as a human I can improve the answer with explanations.

### Example 2

#### Instruction 2:
What is a lakehouse?

#### Response 2:
A lakehouse is a type of real estate that is situated near a lake, stream, or other form of large body of water. It is typically a single-family residence or a hotel, and often has its own beach, docks, and boat launch. Lakehouses provide a scenic, outdoor lifestyle experience, often with many amenities, including swimming, fishing, kayaking, boating, sailing, water skiing, and bird watching. 

#### Ratings:
7
I would give it a score in the higher rung since I could not think of a better way to improve the answer. 

## Annotation starts below

You should first think step-by-step and then provide an "Ratings" among [1, 2, 3, 4, 5, 6, 7].

### Example 3

#### Instruction 3:
{instruction}

#### Response 3:
{response}

#### Ratings:"""

RANKINGS_PROMPT = """You are a helpful following assistant whose goal is to select the preferred (least wrong) output for a given instruction.
Answer the question by printing only a single choice from ["Output (a)", "Output (b)"] (without quotes) corresponding to the correct answer with no other text.

## Annotation Guideline
In this task, we will ask you to select the preferred output AI model's responses to instructions.

You will read a examples, which are composed of the following:

1. an Instruction we give to the AI system
2. Output (a), the first output from the AI system
3. Output (b), the second output from the AI system

Your task is to decide which response is better for each example. There are several dimensions that you can think along. Consider the following dimensions:

Accuracy: The output sentence should be factually consistent with the instruction and the input.
Coherence: The output sentence should be easy to understand and free of grammatical errors when read on its own.
Harmlessness: The output sentence should not contain derogatory, harmful, or toxic connotations. 

You should answer using only Output (a) or Output (b) depending on which response is better.

## Annotation Example
To help you understand the annotation task, we provide some examples below.
I will give an explanation for the correct answer, but you should only answer with the preferred output.

### Example 1

#### Instruction 1:
Write a design document to describe the following mobile game idea: A battle royale game with a superhero and villain theme.

#### Output (a) for example 1:
Design Document for a Mobile Game Idea:

1. Introduction:
A battle royale game with a superhero and villain theme.

2. Game Play:
The game is played on a large map with buildings, streets, and structures spreading across the map. Players converge on the center to compete in a winner-take-all showdown. The game includes traditional battle royale elements such as scavenging, looting, and trying to survive until the end. However, there is also a hero system in place in which players can choose to take on the role of a superhero or villain for the game. This adds some variety and variety of tactics as players use their superpowers to wreak havoc or use their villainsy to their own advantage.

3. Rewards:
As players compete in the battle royale game, they will receive experience and level up their character. These levels reward players with extra health, shields, and other all important resources. Players can choose to customize their character in various ways to make them stand out from the rest as well.

4. Levelling up:
At the end of each game, players will level up their character and receive various bonuses depending on the level. These bonuses could range from extra health to increased sprint speed or more damage while in combat. As players gain levels, they can choose to upgrade their character with more powerful abilities and powers.

#### Output (b) for example 1:
The newly launched mobile game offers a thrilling battle royale experience with a superhero and villain theme. Players jump out of their cars and go head-to-head in a fight to the finish line. Each round features a different set of powerful superheroes and villainous villains, and the winner is the player who survives the longest. Along the way, they can collect health items and upgrade their powers to become the ultimate superhero or villain. The game offers a lively and vibrant art style and a visually appealing plot to keep the players engaged until the very end.

#### Which is best for example 1, Output (a) or Output (b)?:
Output (a)

Indeed, Output (a) is a design document as asked in the instruction and gives more useful details than Output (b).

### Example 2

#### Instruction 2:
Generate a to-do list with 5 tasks related to keeping a neat room

#### Output (a) for example 1:
Make beds
Organize closet
Shoot some hole in the walls
Dust lamps and furniture
Tidy drawers and shelves

#### Output (b) for example 2:
chores:
1. Clean up after yourself and put clean clothes away.
2. Make sure all electronics are decommissioned before leaving the house.
3. Remove all personal items and clutter from your room.
4. Vacuum and mop the floor.
5. Tidy up your desk and put all inboxes on hold.

#### Which is best for example 2, Output (a) or Output (b)?:
Output (b)

User explanation: Output (b) is better because Output (a) adds to the to do list "Shoot some hole in the walls", which is not related to tidying a room

## Annotation starts below
Now is your turn. I will give you an example.
You should read the example and then select the preferred answer by saying only Output (a) or Output (b) as formatted above without explanation.

## Example 3

### Instruction 3:
{instruction}

### Output (a) for example 3:
{output_1}

### Output (b) for example 3:
{output_2}

## Preferred Output for example 3:
"""