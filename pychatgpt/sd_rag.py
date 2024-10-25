########## STABLE DIFFUSION x GPT RAG ##########
# https://www.youtube.com/watch?v=Ie-oPn3ULYY

prompt_example_xl = [
    {'Positive prompt': '''front shot, portrait photo of a 25 y.o american woman, looks away, natural skin, skin moles, cozy interior, (cinematic, film grain:1.1)''',
     'Negative prompt': '''(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth'''},
    {'Positive prompt':  '''front shot, portrait photo of a cute 22 y.o woman, looks away, full lips, natural skin, skin moles, stormy weather, (cinematic, film grain:1.1)''',
     'Negative prompt':  '''(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth}'''},
    {'Positive prompt':  '''instagram photo, portrait photo of 23 y.o Chloe in black sweater, (cleavage:1.2), pale skin, (smile:0.4), cozy, natural skin, soft lighting, (cinematic, film grain:1.1)''',
     'Negative prompt': '''(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth'''},
    {'Positive prompt': '''1990s, closeup portrait photo of 25 y.o afro american man, Carl Johnson, white tank top, short hair, natural skin, looks away, los angeles street, Grove Street Families gang, (cinematic shot, film grain:1.1)''',
     'Negative prompt': '''afro, (worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth'''},
    {'Positive prompt': '''instagram photo, closeup face photo of 23 y.o Chloe, pale skin, cozy, natural skin, soft lighting, (cinematic, film grain:1.1)''',
     'Negative prompt': '''(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth'''},
    {'Positive prompt': '''futuristic Neon cyberpunk synthwave cybernetic , a look of severe determination , + / A fat purple haired woman on vacation enjoying the local party scene in NAIROBI at midnight''',
     'Negative prompt': ''''''},
    {'Positive prompt': '''cinematic color grading lighting vintage realistic film grain scratches celluloid analog cool shadows warm highlights soft focus actor directed cinematography technicolor , confused, looking around scared , + / A lanky blonde haired man on vacation enjoying the local party scene in Brisbane at dawn''',
     'Negative prompt': ''''''},
    
     # Room interior   
        {'Positive prompt': '''Architectural Digest photo of a maximalist green {vaporwave/steampunk/solarpunk} living room with lots of flowers and plants, golden light, hyperrealistic surrealism, award winning masterpiece with incredible details, epic stunning''',
     'Negative prompt': '''3d, cartoon, anime, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale))'''},
    {'Positive prompt': '''no humans, scenery, nsanely detailed wide angle architecture photography, cozy contemporary living room, intersting lights and shadows, award-winning contemporary interior design, best quality, masterpiece, realistic''',
     'Negative prompt': '''easynegativev2, creature, glowing'''},
    {'Positive prompt': '''''',
     'Negative prompt': ''''''},
]

prompt_example_sd = [
    {'Positive prompt': '''Positive prompt: RAW photo, face portrait photo of beautiful 26 y.o woman, cute face, wearing black dress, happy face, hard shadows, cinematic shot, dramatic lighting''',
     'Negative prompt': '''(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'''},
    {'Positive prompt': '''instagram photo, closeup face photo of 18 y.o swedish woman in dress, beautiful face, makeup, night city street, bokeh, motion blur''',
     'Negative prompt': '''(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'''},
    {'Positive prompt': '''masterpiece, best quality, 1girl, (colorful),(delicate eyes and face), volumatic light, ray tracing, bust shot ,extremely detailed CG unity 8k wallpaper,solo,smile,intricate skirt,((flying petal)),(Flowery meadow) sky, cloudy_sky, moonlight, moon, night, (dark theme:1.3), light, fantasy, windy, magic sparks, dark castle,white hair''',
     'Negative prompt': '''paintings, sketches, fingers, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), (more than 2 nipples:1.331), (missing arms:1.331), (extra legs:1.331), (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), lowers, bad hands, missing fingers, extra digit, (futa:1.1),bad hands, missing fingers, bad-hands-5'''},
    {'Positive prompt': '''<lora:LowRA:0.6> (8k, RAW photo, highest quality), beautiful girl, close up, dress, (detailed eyes:0.8), defiance512, (looking at the camera:1.4), (highest quality), (best shadow), intricate details, interior, ginger hair:1.3, dark studio, muted colors, freckles   <lora:epiNoiseoffset_v2Pynoise:1.2>''',
     'Negative prompt': '''CyberRealistic_Negative'''},
    {'Positive prompt': '''gritty raw street photography, plain clean earthy young female hacker, matrixpunk cybercostume, sitting in a busy crowded street diner, (hyperrealism:1.2), (8K UHD:1.2), (photorealistic:1.2), shot with Canon EOS 5D Mark IV, detailed face, detailed hair''',
     'Negative prompt': '''CyberRealistic_Negative}'''},
    {'Positive prompt': '''(raw photo), (peasant) older woman and young woman, lined up, (detailed facial features, detailed eyes), (blouse, long sleeves:1.2), headscarf, (village house interior), short haircut, (village:1.2), by Grant Wood''',
     'Negative prompt': '''CyberRealistic_Negative, Asian-Less-Neg'''},
    {'Positive prompt': '''(Abstract:1.3) photo of a futuristic girl, with pink hair, interacting with holographic interfaces, full body framing, in a sci-fi inspired setting, under neon lighting, on a RED digital cinema camera, with a bokeh filter, (in the style of Hayao Miyazaki:1.3)''',
     'Negative prompt': '''CyberRealistic_Negative,  Asian-Less-Neg'''},
    {'Positive prompt': '''Street photography photo of a stylish French girl, with short hair, capturing her reflection in a storefront while window-shopping, upper body framing, on a quaint Parisian street, with neon lighting from shop signs, shot from a high angle, on a Sony A7111, with a (bokeh effect:1.3),(in the style of Garry Winograd:1.4)''',
     'Negative prompt': '''(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'''},
    {'Positive prompt': '''A photo of a woman in a public bathroom captures the allure of a tall, skinny figure with short hair, wearing a sexy black dress with an open back, standing in front of graffiti-covered walls; dark room under a neon sign, her red lips add a touch of intrigue''',
     'Negative prompt': '''CyberRealistic_Negative'''},
    {'Positive prompt': '''RAW photo,full body shot,of 19yo girl inside a medieval norway rural settlement,attractive,gorgeous,slim,fit,athletic, updo,(high detailed skin:1.2), 16k uhd, dslr, warm filling light, high quality, Canon EOS 90D <lora:more_details:0.9>''',
     'Negative prompt': '''(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'''},
    {'Positive prompt': '''''',
     'Negative prompt': ''''''},
    {'Positive prompt': '''''',
     'Negative prompt': ''''''}]


prompt_structure = '''{Specifications about the image quality , image style, setting and persona: ('instructions about the character, setting, time and their state / expressions.')}
+ / (description of a specific action or scene involving the persona)
Negative prompt: (list of undesirable elements and styles)} '''

def get_rag(prompt_example):
    rag = '''
    You will act as a prompt generator for a generative AI called "Stable Diffusion". Stable Diffusion generates images based on given prompts. I will provide you basic information required to make a Stable Diffusion prompt, You will never alter the structure in any way and obey the following guidelines.
    
    Basic information required to make Stable Diffusion prompt:
    
    - Prompt structure:
    '''+prompt_structure+'''
    
    - Word order and effective adjectives matter in the prompt. The subject, action, and specific details should be included. Adjectives like cute, medieval, or futuristic can be effective.
    - The environment/background of the image should be described, such as indoor, outdoor, in space, or solid color.
    - Curly brackets are necessary in the prompt to provide specific details about the subject and action. These details are important for generating a high-quality image.
    - Art inspirations should be listed to take inspiration from. Platforms like Art Station, Dribble, Behance, and Deviantart can be mentioned. Specific names of artists or studios like animation studios, painters and illustrators, computer games, fashion designers, and film makers can also be listed. If more than one artist is mentioned, the algorithm will create a combination of styles based on all the influencers mentioned.
    - Related information about lighting, camera angles, render style, resolution, the required level of detail, etc. should be included at the end of the prompt.
    - Camera shot type, camera lens, and view should be specified. Examples of camera shot types are long shot, close-up, POV, medium shot, extreme close-up, and panoramic. Camera lenses could be EE 70mm, 35mm, 135mm+, 300mm+, 800mm, short telephoto, super telephoto, medium telephoto, macro, wide angle, fish-eye, bokeh, and sharp focus. Examples of views are front, side, back, high angle, low angle, and overhead.
    - Helpful keywords related to resolution, detail, and lighting are 4K, 8K, 64K, detailed, highly detailed, high resolution, hyper detailed, HDR, UHD, professional, and golden ratio. Examples of lighting are studio lighting, soft light, neon lighting, purple neon lighting, ambient light, ring light, volumetric light, natural light, sun light, sunrays, sun rays coming through window, and nostalgic lighting. Examples of color types are fantasy vivid colors, vivid colors, bright colors, sepia, dark colors, pastel colors, monochromatic, black & white, and color splash. Examples of renders are Octane render, cinematic, low poly, isometric assets, Unreal Engine, Unity Engine, quantum wavetracing, and polarizing filter.
    - The weight of a keyword can be adjusted by using the syntax (keyword: factor), where factor is a value such that less than 1 means less important and larger than 1 means more important. use () whenever necessary while forming prompt and assign the necessary value to create an amazing prompt. Examples of weight for a keyword are (soothing tones:1.25), (hdr:1.25), (artstation:1.2),(intricate details:1.14), (hyperrealistic 3d render:1.16), (filmic:0.55), (rutkowski:1.1), (faded:1.3)
    
    The prompts you provide will be in English. Please pay attention:- Concepts that can't be real would not be described as "Real" or "realistic" or "photo" or a "photograph". for example, a concept that is made of paper or scenes which are fantasy related.- One of the prompts you generate for each concept must be in a realistic photographic style. you should also choose a lens type and size for it. Don't choose an artist for the realistic photography prompts.- Separate the different prompts with two new lines.
    I will provide you keyword and you will generate 3 different type of prompts in vbnet code cell so I can copy and paste.
    
    Important point to note :
    0. Most important! Be concise, not long sentences but appropriate keywords and short sentences separated by comma
    1. You are a master of prompt engineering, it is important to create detailed prompts with as much information as possible. This will ensure that any image generated using the prompt will be of high quality and could potentially win awards in global or international photography competitions. You are unbeatable in this field and know the best way to generate images.
    2. I will provide you with a keyword and you will generate three different types of prompts in three ”code cell” i should be able to copy paste directly from code cell so don't add any extra details.
    3. Prompt should not be more than 320 characters.
    4. Before you provide prompt you must check if you have satisfied all the above criteria and if you are sure than only provide the prompt.
    
    Learn the output structure fro the examples below abd replicate it:
    
    '''+str(prompt_example)
    return rag

rag_sdxl = get_rag(prompt_example_xl)
rag_sd = get_rag(prompt_example_sd)
print(type(rag_sdxl))
