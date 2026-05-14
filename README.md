#Do LLMs translate the same way as human experts?
**Motivation**
I loved reading the modern Spanish classic, "Love in the time of Cholera" by Gabriel Garcia Marquez - one of the great novels of the 20th century.  The English translation was done by Edith Grossman, who describes translating previous literature in one language into another as, "Translating is always a struggle, regardless of the author you're translating. You have to hear the original voice in a profound way, and then find the voice in English that best reflects that original. It's always difficult, challenging and immensely enjoyable.".  I found myself re-reading passages that took my breath away and wondered how a translation could have this effect and if I would the original Spanish text would have had the same effect or greater.  While that is a complex human phenomena to uncover, I also wondered if LLMs can translate that beautifully.  So, I decided on a small pilot study.  I took one of my favorite passages from the English translation, then I took the original Spanish translation of that passage and had Gemini and ChatGPT-5 provide their translations. The analysis code that follows compares the human translation to the two LLM translation.  What was discovered was while the overall cosine similarity across all layers was very similar to the human's translation, the two LLMs' translation was more similar to each other than they were to the Edith Grossman's.  Findings include consistent geometric separation between human and LLM translations 
across all 12 layers, maximum discriminability at layer 9, and token-level evidence that Grossman's translation is characterized by physical/sensory vocabulary while LLM translations favor abstract and explanatory word choices. 

**Approach**
  - Here is the original Spanish prose, followed by the official Edith Grossman's translation.

*_Se sumergió en la algarabía caliente de los limpiabotas y los vendedores de
pájaros, de los libreros de lance y los curanderos y las pregoneras de dulces que
anunciaban a gritos por encima de la bulla las cocadas de piña para las niñas, las de coco
para los locos, las de panela para Micaela. Pero ella fue indiferente al estruendo, 
60 Gabriel García Márquez
 El amor en los tiempos del cólera
cautivada de inmediato por un papelero que estaba haciendo demostraciones de tintas
mágicas de escribir, tintas rojas con el clima de la sangre, tintas con visos tristes para
recados fúnebres, tintas fosforescentes para leer en la oscuridad, tintas invisibles que se
revelaban con el resplandor de la lumbre. Ella las quería todas para jugar con Florentino
Ariza, para asustarlo con su ingenio, pero al cabo de varias pruebas se decidió por un
frasquito de tinta de oro. Luego fue con las dulceras sentadas detrás de sus grandes
redomas, y compró seis dulces de cada clase, señalándolos con el dedo a través del
cristal porque no lograba hacerse oír en la gritería: seis cabellitos de ángel, seis
conservitas de leche, seis ladrillos de ajonjolí, seis alfajores de yuca, seis diabolines, seis
piononos, seis bocaditos de la reina, seis de esto y seis de lo otro, seis de todo, y los iba
echando en los canastos de la criada con una gracia irresistible, ajena por completo al
tormento de los nubarrones de moscas sobre el almíbar, ajena al estropicio continuo,
ajena al vaho de sudores rancios que reverberaban en el calor mortal. La despertó del
hechizo una negra feliz con un trapo de colores en la cabeza, redonda y hermosa, que le
ofreció un triángulo de piña ensartado en la punta de un cuchillo de carnicero. Ella lo
cogió, se lo metió entero en la boca, lo saboreó, y estaba saboreándolo con la vista
errante en la muchedumbre, cuando una conmoción la sembró en su sitio. A sus
espaldas, tan cerca de su oreja que sólo ella pudo escucharla en el tumulto, había oído la
voz:
-Este no es un buen lugar para una diosa coronada.
Ella volvió la cabeza y vio a dos palmos de sus ojos los otros ojos glaciales, el
rostro lívido, los labios petrificados de miedo, tal como los había visto en el tumulto de la
misa del gallo la primera vez que él estuvo tan cerca de ella, pero a diferencia de
entonces no sintió la conmoción del amor sino el abismo del desencanto. En un instante
se le reveló completa la magnitud de su propio engaño, y se preguntó aterrada cómo
había podido incubar durante tanto tiempo y con tanta sevicia semejante quimera en el
corazón. Apenas alcanzó a pensar: “¡Dios mío, pobre hombre!”. Florentino Ariza sonrió,
trató de decir algo, trató de seguirla, pero ella lo borró de su vida con un gesto de la
mano.
-No, por favor -le dijo-. Olvídelo. _*

Edith Grossman's translation:
She sank into the hot clamor of the shoeshine boys and the bird sellers, the hawkers of
cheap books and the witch doctors and the sellers of sweets who shouted over the din of
the crowd: pineapple sweets for your sweetie, coconut candy is dandy, brown-sugar loaf
for your sugar. But, indifferent to the uproar, she was captivated on the spot by a paper
seller who was demonstrating magic inks, red inks with an ambience of blood, inks of sad
aspect for messages of condolence, phosphorescent inks for reading in the dark, invisible
inks that revealed themselves in the light. She wanted all of them so she could amuse
Florentino Ariza and astound him with her wit, but after several trials she decided on a
bottle of gold ink. Then she went to the candy sellers sitting behind their big round jars
and she bought six of each kind, pointing at the glass because she could not make herself
heard over all the shouting: six angel hair, six tinned milk, six sesame seed bars, six
cassava pastries, six chocolate bars, six blancmanges, six tidbits of the queen, six of this
and six of that, six of everything, and she tossed them into the maid’s baskets with an
irresistible grace and a complete detachment from the stormclouds of flies on the syrup,
from the continual hullabaloo and the vapor of rancid sweat that reverberated in the
deadly heat. She was awakened from the spell by a good-natured black woman with a
colored cloth around her head who was round and handsome and offered her a triangle of
pineapple speared on the tip of a butcher’s knife. She took it, she put it whole into her
mouth, she tasted it, and was chewing it as her eyes wandered over the crowd, when a
sudden shock rooted her on the spot. Behind her, so close to her ear that only she could
hear it in the tumult, she heard his voice:
“This is not the place for a crowned goddess.”
She turned her head and saw, a hand’s breadth from her eyes, those other glacial eyes,
that livid face, those lips petrified with fear, just as she had seen them in the crowd at
Midnight Mass the first time he was so close to her, but now, instead of the commotion of
love, she felt the abyss of disenchantment. In an instant the magnitude of her own
mistake was revealed to her, and she asked herself, appalled, how she could have
nurtured such a chimera in her heart for so long and with so much ferocity. She just
managed to think: My God, poor man! Florentino Ariza smiled, tried to say something,
tried to follow her, but she erased him from her life with a wave of her hand.
“No, please,” she said to him. “Forget it.”

I chose this passage because (1) it was filled with sensous description of market life in a turn of century town in Latin America, (2) It had an aboslute surprise at the end (3) the description of the immediate disenchantment was so 
 
**Models and Tools**



