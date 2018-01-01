/*
 * Copyright (c) 2018 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.models;

import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.zip.ZipFile;

/**
 * The type Vgg 16.
 */
public abstract class VGG16 extends ImageClassifier {
  
  /**
   * From s 3 vgg 16 hdf 5.
   *
   * @return the vgg 16 hdf 5
   */
  public static VGG16 fromS3_HDF5() {
    try {
      return new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5"))));
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * From zip vgg 16.
   *
   * @param file the file
   * @return the vgg 16
   */
  public static VGG16 fromZip(File file) {
    try {
      return new VGG16_Zip(NNLayer.fromZip(new ZipFile(file)));
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
  }
  
  @Override
  public Tensor prefilter(Tensor tensor) {
    tensor = tensor.mapCoords(getBiasFunction(tensor));
    tensor = tensor.mapCoords(getPermuteFunction(tensor));
    return tensor.permuteDimensions(0, 1, 2);
  }
  
  /**
   * Gets bias function.
   *
   * @param tensor the tensor
   * @return the bias function
   */
  public ToDoubleFunction<Coordinate> getBiasFunction(Tensor tensor) {
    return c1 -> {
      if (c1.getCoords()[2] == 0) return tensor.get(c1) - 103.939;
      if (c1.getCoords()[2] == 1) return tensor.get(c1) - 116.779;
      if (c1.getCoords()[2] == 2) return tensor.get(c1) - 123.68;
      else return tensor.get(c1);
    };
  }
  
  /**
   * Gets permute function.
   *
   * @param tensor the tensor
   * @return the permute function
   */
  public ToDoubleFunction<Coordinate> getPermuteFunction(Tensor tensor) {
    return c -> {
      if (c.getCoords()[2] == 0) return tensor.get(c.getCoords()[0], c.getCoords()[1], 0);
      if (c.getCoords()[2] == 1) return tensor.get(c.getCoords()[0], c.getCoords()[1], 1);
      if (c.getCoords()[2] == 2) return tensor.get(c.getCoords()[0], c.getCoords()[1], 2);
      else throw new RuntimeException();
    };
  }
  
  @Override
  public List<String> getCategories() {
    return Arrays.stream((
                           "tench, Tinca tinca\n" +
                             "goldfish, Carassius auratus\n" +
                             "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias\n" +
                             "tiger shark, Galeocerdo cuvieri\n" +
                             "hammerhead, hammerhead shark\n" +
                             "electric ray, crampfish, numbfish, torpedo\n" +
                             "stingray\n" +
                             "cock\n" +
                             "hen\n" +
                             "ostrich, Struthio camelus\n" +
                             "brambling, Fringilla montifringilla\n" +
                             "goldfinch, Carduelis carduelis\n" +
                             "house finch, linnet, Carpodacus mexicanus\n" +
                             "junco, snowbird\n" +
                             "indigo bunting, indigo finch, indigo bird, Passerina cyanea\n" +
                             "robin, American robin, Turdus migratorius\n" +
                             "bulbul\n" +
                             "jay\n" +
                             "magpie\n" +
                             "chickadee\n" +
                             "water ouzel, dipper\n" +
                             "kite\n" +
                             "bald eagle, American eagle, Haliaeetus leucocephalus\n" +
                             "vulture\n" +
                             "great grey owl, great gray owl, Strix nebulosa\n" +
                             "European fire salamander, Salamandra salamandra\n" +
                             "common newt, Triturus vulgaris\n" +
                             "eft\n" +
                             "spotted salamander, Ambystoma maculatum\n" +
                             "axolotl, mud puppy, Ambystoma mexicanum\n" +
                             "bullfrog, Rana catesbeiana\n" +
                             "tree frog, tree-frog\n" +
                             "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui\n" +
                             "loggerhead, loggerhead turtle, Caretta caretta\n" +
                             "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea\n" +
                             "mud turtle\n" +
                             "terrapin\n" +
                             "box turtle, box tortoise\n" +
                             "banded gecko\n" +
                             "common iguana, iguana, Iguana iguana\n" +
                             "American chameleon, anole, Anolis carolinensis\n" +
                             "whiptail, whiptail lizard\n" +
                             "agama\n" +
                             "frilled lizard, Chlamydosaurus kingi\n" +
                             "alligator lizard\n" +
                             "Gila monster, Heloderma suspectum\n" +
                             "green lizard, Lacerta viridis\n" +
                             "African chameleon, Chamaeleo chamaeleon\n" +
                             "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis\n" +
                             "African crocodile, Nile crocodile, Crocodylus niloticus\n" +
                             "American alligator, Alligator mississipiensis\n" +
                             "triceratops\n" +
                             "thunder snake, worm snake, Carphophis amoenus\n" +
                             "ringneck snake, ring-necked snake, ring snake\n" +
                             "hognose snake, puff adder, sand viper\n" +
                             "green snake, grass snake\n" +
                             "king snake, kingsnake\n" +
                             "garter snake, grass snake\n" +
                             "water snake\n" +
                             "vine snake\n" +
                             "night snake, Hypsiglena torquata\n" +
                             "boa constrictor, Constrictor constrictor\n" +
                             "rock python, rock snake, Python sebae\n" +
                             "Indian cobra, Naja naja\n" +
                             "green mamba\n" +
                             "sea snake\n" +
                             "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus\n" +
                             "diamondback, diamondback rattlesnake, Crotalus adamanteus\n" +
                             "sidewinder, horned rattlesnake, Crotalus cerastes\n" +
                             "trilobite\n" +
                             "harvestman, daddy longlegs, Phalangium opilio\n" +
                             "scorpion\n" +
                             "black and gold garden spider, Argiope aurantia\n" +
                             "barn spider, Araneus cavaticus\n" +
                             "garden spider, Aranea diademata\n" +
                             "black widow, Latrodectus mactans\n" +
                             "tarantula\n" +
                             "wolf spider, hunting spider\n" +
                             "tick\n" +
                             "centipede\n" +
                             "black grouse\n" +
                             "ptarmigan\n" +
                             "ruffed grouse, partridge, Bonasa umbellus\n" +
                             "prairie chicken, prairie grouse, prairie fowl\n" +
                             "peacock\n" +
                             "quail\n" +
                             "partridge\n" +
                             "African grey, African gray, Psittacus erithacus\n" +
                             "macaw\n" +
                             "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita\n" +
                             "lorikeet\n" +
                             "coucal\n" +
                             "bee eater\n" +
                             "hornbill\n" +
                             "hummingbird\n" +
                             "jacamar\n" +
                             "toucan\n" +
                             "drake\n" +
                             "red-breasted merganser, Mergus serrator\n" +
                             "goose\n" +
                             "black swan, Cygnus atratus\n" +
                             "tusker\n" +
                             "echidna, spiny anteater, anteater\n" +
                             "platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus\n" +
                             "wallaby, brush kangaroo\n" +
                             "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus\n" +
                             "wombat\n" +
                             "jellyfish\n" +
                             "sea anemone, anemone\n" +
                             "brain coral\n" +
                             "flatworm, platyhelminth\n" +
                             "nematode, nematode worm, roundworm\n" +
                             "conch\n" +
                             "snail\n" +
                             "slug\n" +
                             "sea slug, nudibranch\n" +
                             "chiton, coat-of-mail shell, sea cradle, polyplacophore\n" +
                             "chambered nautilus, pearly nautilus, nautilus\n" +
                             "Dungeness crab, Cancer magister\n" +
                             "rock crab, Cancer irroratus\n" +
                             "fiddler crab\n" +
                             "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica\n" +
                             "American lobster, Northern lobster, Maine lobster, Homarus americanus\n" +
                             "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish\n" +
                             "crayfish, crawfish, crawdad, crawdaddy\n" +
                             "hermit crab\n" +
                             "isopod\n" +
                             "white stork, Ciconia ciconia\n" +
                             "black stork, Ciconia nigra\n" +
                             "spoonbill\n" +
                             "flamingo\n" +
                             "little blue heron, Egretta caerulea\n" +
                             "American egret, great white heron, Egretta albus\n" +
                             "bittern\n" +
                             "crane\n" +
                             "limpkin, Aramus pictus\n" +
                             "European gallinule, Porphyrio porphyrio\n" +
                             "American coot, marsh hen, mud hen, water hen, Fulica americana\n" +
                             "bustard\n" +
                             "ruddy turnstone, Arenaria interpres\n" +
                             "red-backed sandpiper, dunlin, Erolia alpina\n" +
                             "redshank, Tringa totanus\n" +
                             "dowitcher\n" +
                             "oystercatcher, oyster catcher\n" +
                             "pelican\n" +
                             "king penguin, Aptenodytes patagonica\n" +
                             "albatross, mollymawk\n" +
                             "grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus\n" +
                             "killer whale, killer, orca, grampus, sea wolf, Orcinus orca\n" +
                             "dugong, Dugong dugon\n" +
                             "sea lion\n" +
                             "Chihuahua\n" +
                             "Japanese spaniel\n" +
                             "Maltese dog, Maltese terrier, Maltese\n" +
                             "Pekinese, Pekingese, Peke\n" +
                             "Shih-Tzu\n" +
                             "Blenheim spaniel\n" +
                             "papillon\n" +
                             "toy terrier\n" +
                             "Rhodesian ridgeback\n" +
                             "Afghan hound, Afghan\n" +
                             "basset, basset hound\n" +
                             "beagle\n" +
                             "bloodhound, sleuthhound\n" +
                             "bluetick\n" +
                             "black-and-tan coonhound\n" +
                             "Walker hound, Walker foxhound\n" +
                             "English foxhound\n" +
                             "redbone\n" +
                             "borzoi, Russian wolfhound\n" +
                             "Irish wolfhound\n" +
                             "Italian greyhound\n" +
                             "whippet\n" +
                             "Ibizan hound, Ibizan Podenco\n" +
                             "Norwegian elkhound, elkhound\n" +
                             "otterhound, otter hound\n" +
                             "Saluki, gazelle hound\n" +
                             "Scottish deerhound, deerhound\n" +
                             "Weimaraner\n" +
                             "Staffordshire bullterrier, Staffordshire bull terrier\n" +
                             "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier\n" +
                             "Bedlington terrier\n" +
                             "Border terrier\n" +
                             "Kerry blue terrier\n" +
                             "Irish terrier\n" +
                             "Norfolk terrier\n" +
                             "Norwich terrier\n" +
                             "Yorkshire terrier\n" +
                             "wire-haired fox terrier\n" +
                             "Lakeland terrier\n" +
                             "Sealyham terrier, Sealyham\n" +
                             "Airedale, Airedale terrier\n" +
                             "cairn, cairn terrier\n" +
                             "Australian terrier\n" +
                             "Dandie Dinmont, Dandie Dinmont terrier\n" +
                             "Boston bull, Boston terrier\n" +
                             "miniature schnauzer\n" +
                             "giant schnauzer\n" +
                             "standard schnauzer\n" +
                             "Scotch terrier, Scottish terrier, Scottie\n" +
                             "Tibetan terrier, chrysanthemum dog\n" +
                             "silky terrier, Sydney silky\n" +
                             "soft-coated wheaten terrier\n" +
                             "West Highland white terrier\n" +
                             "Lhasa, Lhasa apso\n" +
                             "flat-coated retriever\n" +
                             "curly-coated retriever\n" +
                             "golden retriever\n" +
                             "Labrador retriever\n" +
                             "Chesapeake Bay retriever\n" +
                             "German short-haired pointer\n" +
                             "vizsla, Hungarian pointer\n" +
                             "English setter\n" +
                             "Irish setter, red setter\n" +
                             "Gordon setter\n" +
                             "Brittany spaniel\n" +
                             "clumber, clumber spaniel\n" +
                             "English springer, English springer spaniel\n" +
                             "Welsh springer spaniel\n" +
                             "cocker spaniel, English cocker spaniel, cocker\n" +
                             "Sussex spaniel\n" +
                             "Irish water spaniel\n" +
                             "kuvasz\n" +
                             "schipperke\n" +
                             "groenendael\n" +
                             "malinois\n" +
                             "briard\n" +
                             "kelpie\n" +
                             "komondor\n" +
                             "Old English sheepdog, bobtail\n" +
                             "Shetland sheepdog, Shetland sheep dog, Shetland\n" +
                             "collie\n" +
                             "Border collie\n" +
                             "Bouvier des Flandres, Bouviers des Flandres\n" +
                             "Rottweiler\n" +
                             "German shepherd, German shepherd dog, German police dog, alsatian\n" +
                             "Doberman, Doberman pinscher\n" +
                             "miniature pinscher\n" +
                             "Greater Swiss Mountain dog\n" +
                             "Bernese mountain dog\n" +
                             "Appenzeller\n" +
                             "EntleBucher\n" +
                             "boxer\n" +
                             "bull mastiff\n" +
                             "Tibetan mastiff\n" +
                             "French bulldog\n" +
                             "Great Dane\n" +
                             "Saint Bernard, St Bernard\n" +
                             "Eskimo dog, husky\n" +
                             "malamute, malemute, Alaskan malamute\n" +
                             "Siberian husky\n" +
                             "dalmatian, coach dog, carriage dog\n" +
                             "affenpinscher, monkey pinscher, monkey dog\n" +
                             "basenji\n" +
                             "pug, pug-dog\n" +
                             "Leonberg\n" +
                             "Newfoundland, Newfoundland dog\n" +
                             "Great Pyrenees\n" +
                             "Samoyed, Samoyede\n" +
                             "Pomeranian\n" +
                             "chow, chow chow\n" +
                             "keeshond\n" +
                             "Brabancon griffon\n" +
                             "Pembroke, Pembroke Welsh corgi\n" +
                             "Cardigan, Cardigan Welsh corgi\n" +
                             "toy poodle\n" +
                             "miniature poodle\n" +
                             "standard poodle\n" +
                             "Mexican hairless\n" +
                             "timber wolf, grey wolf, gray wolf, Canis lupus\n" +
                             "white wolf, Arctic wolf, Canis lupus tundrarum\n" +
                             "red wolf, maned wolf, Canis rufus, Canis niger\n" +
                             "coyote, prairie wolf, brush wolf, Canis latrans\n" +
                             "dingo, warrigal, warragal, Canis dingo\n" +
                             "dhole, Cuon alpinus\n" +
                             "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus\n" +
                             "hyena, hyaena\n" +
                             "red fox, Vulpes vulpes\n" +
                             "kit fox, Vulpes macrotis\n" +
                             "Arctic fox, white fox, Alopex lagopus\n" +
                             "grey fox, gray fox, Urocyon cinereoargenteus\n" +
                             "tabby, tabby cat\n" +
                             "tiger cat\n" +
                             "Persian cat\n" +
                             "Siamese cat, Siamese\n" +
                             "Egyptian cat\n" +
                             "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor\n" +
                             "lynx, catamount\n" +
                             "leopard, Panthera pardus\n" +
                             "snow leopard, ounce, Panthera uncia\n" +
                             "jaguar, panther, Panthera onca, Felis onca\n" +
                             "lion, king of beasts, Panthera leo\n" +
                             "tiger, Panthera tigris\n" +
                             "cheetah, chetah, Acinonyx jubatus\n" +
                             "brown bear, bruin, Ursus arctos\n" +
                             "American black bear, black bear, Ursus americanus, Euarctos americanus\n" +
                             "ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus\n" +
                             "sloth bear, Melursus ursinus, Ursus ursinus\n" +
                             "mongoose\n" +
                             "meerkat, mierkat\n" +
                             "tiger beetle\n" +
                             "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle\n" +
                             "ground beetle, carabid beetle\n" +
                             "long-horned beetle, longicorn, longicorn beetle\n" +
                             "leaf beetle, chrysomelid\n" +
                             "dung beetle\n" +
                             "rhinoceros beetle\n" +
                             "weevil\n" +
                             "fly\n" +
                             "bee\n" +
                             "ant, emmet, pismire\n" +
                             "grasshopper, hopper\n" +
                             "cricket\n" +
                             "walking stick, walkingstick, stick insect\n" +
                             "cockroach, roach\n" +
                             "mantis, mantid\n" +
                             "cicada, cicala\n" +
                             "leafhopper\n" +
                             "lacewing, lacewing fly\n" +
                             "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk\n" +
                             "damselfly\n" +
                             "admiral\n" +
                             "ringlet, ringlet butterfly\n" +
                             "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus\n" +
                             "cabbage butterfly\n" +
                             "sulphur butterfly, sulfur butterfly\n" +
                             "lycaenid, lycaenid butterfly\n" +
                             "starfish, sea star\n" +
                             "sea urchin\n" +
                             "sea cucumber, holothurian\n" +
                             "wood rabbit, cottontail, cottontail rabbit\n" +
                             "hare\n" +
                             "Angora, Angora rabbit\n" +
                             "hamster\n" +
                             "porcupine, hedgehog\n" +
                             "fox squirrel, eastern fox squirrel, Sciurus niger\n" +
                             "marmot\n" +
                             "beaver\n" +
                             "guinea pig, Cavia cobaya\n" +
                             "sorrel\n" +
                             "zebra\n" +
                             "hog, pig, grunter, squealer, Sus scrofa\n" +
                             "wild boar, boar, Sus scrofa\n" +
                             "warthog\n" +
                             "hippopotamus, hippo, river horse, Hippopotamus amphibius\n" +
                             "ox\n" +
                             "water buffalo, water ox, Asiatic buffalo, Bubalus bubalis\n" +
                             "bison\n" +
                             "ram, tup\n" +
                             "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis\n" +
                             "ibex, Capra ibex\n" +
                             "hartebeest\n" +
                             "impala, Aepyceros melampus\n" +
                             "gazelle\n" +
                             "Arabian camel, dromedary, Camelus dromedarius\n" +
                             "llama\n" +
                             "weasel\n" +
                             "mink\n" +
                             "polecat, fitch, foulmart, foumart, Mustela putorius\n" +
                             "black-footed ferret, ferret, Mustela nigripes\n" +
                             "otter\n" +
                             "skunk, polecat, wood pussy\n" +
                             "badger\n" +
                             "armadillo\n" +
                             "three-toed sloth, ai, Bradypus tridactylus\n" +
                             "orangutan, orang, orangutang, Pongo pygmaeus\n" +
                             "gorilla, Gorilla gorilla\n" +
                             "chimpanzee, chimp, Pan troglodytes\n" +
                             "gibbon, Hylobates lar\n" +
                             "siamang, Hylobates syndactylus, Symphalangus syndactylus\n" +
                             "guenon, guenon monkey\n" +
                             "patas, hussar monkey, Erythrocebus patas\n" +
                             "baboon\n" +
                             "macaque\n" +
                             "langur\n" +
                             "colobus, colobus monkey\n" +
                             "proboscis monkey, Nasalis larvatus\n" +
                             "marmoset\n" +
                             "capuchin, ringtail, Cebus capucinus\n" +
                             "howler monkey, howler\n" +
                             "titi, titi monkey\n" +
                             "spider monkey, Ateles geoffroyi\n" +
                             "squirrel monkey, Saimiri sciureus\n" +
                             "Madagascar cat, ring-tailed lemur, Lemur catta\n" +
                             "indri, indris, Indri indri, Indri brevicaudatus\n" +
                             "Indian elephant, Elephas maximus\n" +
                             "African elephant, Loxodonta africana\n" +
                             "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens\n" +
                             "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca\n" +
                             "barracouta, snoek\n" +
                             "eel\n" +
                             "coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch\n" +
                             "rock beauty, Holocanthus tricolor\n" +
                             "anemone fish\n" +
                             "sturgeon\n" +
                             "gar, garfish, garpike, billfish, Lepisosteus osseus\n" +
                             "lionfish\n" +
                             "puffer, pufferfish, blowfish, globefish\n" +
                             "abacus\n" +
                             "abaya\n" +
                             "academic gown, academic robe, judge's robe\n" +
                             "accordion, piano accordion, squeeze box\n" +
                             "acoustic guitar\n" +
                             "aircraft carrier, carrier, flattop, attack aircraft carrier\n" +
                             "airliner\n" +
                             "airship, dirigible\n" +
                             "altar\n" +
                             "ambulance\n" +
                             "amphibian, amphibious vehicle\n" +
                             "analog clock\n" +
                             "apiary, bee house\n" +
                             "apron\n" +
                             "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin\n" +
                             "assault rifle, assault gun\n" +
                             "backpack, back pack, knapsack, packsack, rucksack, haversack\n" +
                             "bakery, bakeshop, bakehouse\n" +
                             "balance beam, beam\n" +
                             "balloon\n" +
                             "ballpoint, ballpoint pen, ballpen, Biro\n" +
                             "Band Aid\n" +
                             "banjo\n" +
                             "bannister, banister, balustrade, balusters, handrail\n" +
                             "barbell\n" +
                             "barber chair\n" +
                             "barbershop\n" +
                             "barn\n" +
                             "barometer\n" +
                             "barrel, cask\n" +
                             "barrow, garden cart, lawn cart, wheelbarrow\n" +
                             "baseball\n" +
                             "basketball\n" +
                             "bassinet\n" +
                             "bassoon\n" +
                             "bathing cap, swimming cap\n" +
                             "bath towel\n" +
                             "bathtub, bathing tub, bath, tub\n" +
                             "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon\n" +
                             "beacon, lighthouse, beacon light, pharos\n" +
                             "beaker\n" +
                             "bearskin, busby, shako\n" +
                             "beer bottle\n" +
                             "beer glass\n" +
                             "bell cote, bell cot\n" +
                             "bib\n" +
                             "bicycle-built-for-two, tandem bicycle, tandem\n" +
                             "bikini, two-piece\n" +
                             "binder, ring-binder\n" +
                             "binoculars, field glasses, opera glasses\n" +
                             "birdhouse\n" +
                             "boathouse\n" +
                             "bobsled, bobsleigh, bob\n" +
                             "bolo tie, bolo, bola tie, bola\n" +
                             "bonnet, poke bonnet\n" +
                             "bookcase\n" +
                             "bookshop, bookstore, bookstall\n" +
                             "bottlecap\n" +
                             "bow\n" +
                             "bow tie, bow-tie, bowtie\n" +
                             "brass, memorial tablet, plaque\n" +
                             "brassiere, bra, bandeau\n" +
                             "breakwater, groin, groyne, mole, bulwark, seawall, jetty\n" +
                             "breastplate, aegis, egis\n" +
                             "broom\n" +
                             "bucket, pail\n" +
                             "buckle\n" +
                             "bulletproof vest\n" +
                             "bullet train, bullet\n" +
                             "butcher shop, meat market\n" +
                             "cab, hack, taxi, taxicab\n" +
                             "caldron, cauldron\n" +
                             "candle, taper, wax light\n" +
                             "cannon\n" +
                             "canoe\n" +
                             "can opener, tin opener\n" +
                             "cardigan\n" +
                             "car mirror\n" +
                             "carousel, carrousel, merry-go-round, roundabout, whirligig\n" +
                             "carpenter's kit, tool kit\n" +
                             "carton\n" +
                             "car wheel\n" +
                             "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM\n" +
                             "cassette\n" +
                             "cassette player\n" +
                             "castle\n" +
                             "catamaran\n" +
                             "CD player\n" +
                             "cello, violoncello\n" +
                             "cellular telephone, cellular phone, cellphone, cell, mobile phone\n" +
                             "chain\n" +
                             "chainlink fence\n" +
                             "chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour\n" +
                             "chain saw, chainsaw\n" +
                             "chest\n" +
                             "chiffonier, commode\n" +
                             "chime, bell, gong\n" +
                             "china cabinet, china closet\n" +
                             "Christmas stocking\n" +
                             "church, church building\n" +
                             "cinema, movie theater, movie theatre, movie house, picture palace\n" +
                             "cleaver, meat cleaver, chopper\n" +
                             "cliff dwelling\n" +
                             "cloak\n" +
                             "clog, geta, patten, sabot\n" +
                             "cocktail shaker\n" +
                             "coffee mug\n" +
                             "coffeepot\n" +
                             "coil, spiral, volute, whorl, helix\n" +
                             "combination lock\n" +
                             "computer keyboard, keypad\n" +
                             "confectionery, confectionary, candy store\n" +
                             "container ship, containership, container vessel\n" +
                             "convertible\n" +
                             "corkscrew, bottle screw\n" +
                             "cornet, horn, trumpet, trump\n" +
                             "cowboy boot\n" +
                             "cowboy hat, ten-gallon hat\n" +
                             "cradle\n" +
                             "crane\n" +
                             "crash helmet\n" +
                             "crate\n" +
                             "crib, cot\n" +
                             "Crock Pot\n" +
                             "croquet ball\n" +
                             "crutch\n" +
                             "cuirass\n" +
                             "dam, dike, dyke\n" +
                             "desk\n" +
                             "desktop computer\n" +
                             "dial telephone, dial phone\n" +
                             "diaper, nappy, napkin\n" +
                             "digital clock\n" +
                             "digital watch\n" +
                             "dining table, board\n" +
                             "dishrag, dishcloth\n" +
                             "dishwasher, dish washer, dishwashing machine\n" +
                             "disk brake, disc brake\n" +
                             "dock, dockage, docking facility\n" +
                             "dogsled, dog sled, dog sleigh\n" +
                             "dome\n" +
                             "doormat, welcome mat\n" +
                             "drilling platform, offshore rig\n" +
                             "drum, membranophone, tympan\n" +
                             "drumstick\n" +
                             "dumbbell\n" +
                             "Dutch oven\n" +
                             "electric fan, blower\n" +
                             "electric guitar\n" +
                             "electric locomotive\n" +
                             "entertainment center\n" +
                             "envelope\n" +
                             "espresso maker\n" +
                             "face powder\n" +
                             "feather boa, boa\n" +
                             "file, file cabinet, filing cabinet\n" +
                             "fireboat\n" +
                             "fire engine, fire truck\n" +
                             "fire screen, fireguard\n" +
                             "flagpole, flagstaff\n" +
                             "flute, transverse flute\n" +
                             "folding chair\n" +
                             "football helmet\n" +
                             "forklift\n" +
                             "fountain\n" +
                             "fountain pen\n" +
                             "four-poster\n" +
                             "freight car\n" +
                             "French horn, horn\n" +
                             "frying pan, frypan, skillet\n" +
                             "fur coat\n" +
                             "garbage truck, dustcart\n" +
                             "gasmask, respirator, gas helmet\n" +
                             "gas pump, gasoline pump, petrol pump, island dispenser\n" +
                             "goblet\n" +
                             "go-kart\n" +
                             "golf ball\n" +
                             "golfcart, golf cart\n" +
                             "gondola\n" +
                             "gong, tam-tam\n" +
                             "gown\n" +
                             "grand piano, grand\n" +
                             "greenhouse, nursery, glasshouse\n" +
                             "grille, radiator grille\n" +
                             "grocery store, grocery, food market, market\n" +
                             "guillotine\n" +
                             "hair slide\n" +
                             "hair spray\n" +
                             "half track\n" +
                             "hammer\n" +
                             "hamper\n" +
                             "hand blower, blow dryer, blow drier, hair dryer, hair drier\n" +
                             "hand-held computer, hand-held microcomputer\n" +
                             "handkerchief, hankie, hanky, hankey\n" +
                             "hard disc, hard disk, fixed disk\n" +
                             "harmonica, mouth organ, harp, mouth harp\n" +
                             "harp\n" +
                             "harvester, reaper\n" +
                             "hatchet\n" +
                             "holster\n" +
                             "home theater, home theatre\n" +
                             "honeycomb\n" +
                             "hook, claw\n" +
                             "hoopskirt, crinoline\n" +
                             "horizontal bar, high bar\n" +
                             "horse cart, horse-cart\n" +
                             "hourglass\n" +
                             "iPod\n" +
                             "iron, smoothing iron\n" +
                             "jack-o'-lantern\n" +
                             "jean, blue jean, denim\n" +
                             "jeep, landrover\n" +
                             "jersey, T-shirt, tee shirt\n" +
                             "jigsaw puzzle\n" +
                             "jinrikisha, ricksha, rickshaw\n" +
                             "joystick\n" +
                             "kimono\n" +
                             "knee pad\n" +
                             "knot\n" +
                             "lab coat, laboratory coat\n" +
                             "ladle\n" +
                             "lampshade, lamp shade\n" +
                             "laptop, laptop computer\n" +
                             "lawn mower, mower\n" +
                             "lens cap, lens cover\n" +
                             "letter opener, paper knife, paperknife\n" +
                             "library\n" +
                             "lifeboat\n" +
                             "lighter, light, igniter, ignitor\n" +
                             "limousine, limo\n" +
                             "liner, ocean liner\n" +
                             "lipstick, lip rouge\n" +
                             "Loafer\n" +
                             "lotion\n" +
                             "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system\n" +
                             "loupe, jeweler's loupe\n" +
                             "lumbermill, sawmill\n" +
                             "magnetic compass\n" +
                             "mailbag, postbag\n" +
                             "mailbox, letter box\n" +
                             "maillot\n" +
                             "maillot, tank suit\n" +
                             "manhole cover\n" +
                             "maraca\n" +
                             "marimba, xylophone\n" +
                             "mask\n" +
                             "matchstick\n" +
                             "maypole\n" +
                             "maze, labyrinth\n" +
                             "measuring cup\n" +
                             "medicine chest, medicine cabinet\n" +
                             "megalith, megalithic structure\n" +
                             "microphone, mike\n" +
                             "microwave, microwave oven\n" +
                             "military uniform\n" +
                             "milk can\n" +
                             "minibus\n" +
                             "miniskirt, mini\n" +
                             "minivan\n" +
                             "missile\n" +
                             "mitten\n" +
                             "mixing bowl\n" +
                             "mobile home, manufactured home\n" +
                             "Model T\n" +
                             "modem\n" +
                             "monastery\n" +
                             "monitor\n" +
                             "moped\n" +
                             "mortar\n" +
                             "mortarboard\n" +
                             "mosque\n" +
                             "mosquito net\n" +
                             "motor scooter, scooter\n" +
                             "mountain bike, all-terrain bike, off-roader\n" +
                             "mountain tent\n" +
                             "mouse, computer mouse\n" +
                             "mousetrap\n" +
                             "moving van\n" +
                             "muzzle\n" +
                             "nail\n" +
                             "neck brace\n" +
                             "necklace\n" +
                             "nipple\n" +
                             "notebook, notebook computer\n" +
                             "obelisk\n" +
                             "oboe, hautboy, hautbois\n" +
                             "ocarina, sweet potato\n" +
                             "odometer, hodometer, mileometer, milometer\n" +
                             "oil filter\n" +
                             "organ, pipe organ\n" +
                             "oscilloscope, scope, cathode-ray oscilloscope, CRO\n" +
                             "overskirt\n" +
                             "oxcart\n" +
                             "oxygen mask\n" +
                             "packet\n" +
                             "paddle, boat paddle\n" +
                             "paddlewheel, paddle wheel\n" +
                             "padlock\n" +
                             "paintbrush\n" +
                             "pajama, pyjama, pj's, jammies\n" +
                             "palace\n" +
                             "panpipe, pandean pipe, syrinx\n" +
                             "paper towel\n" +
                             "parachute, chute\n" +
                             "parallel bars, bars\n" +
                             "park bench\n" +
                             "parking meter\n" +
                             "passenger car, coach, carriage\n" +
                             "patio, terrace\n" +
                             "pay-phone, pay-station\n" +
                             "pedestal, plinth, footstall\n" +
                             "pencil box, pencil case\n" +
                             "pencil sharpener\n" +
                             "perfume, essence\n" +
                             "Petri dish\n" +
                             "photocopier\n" +
                             "pick, plectrum, plectron\n" +
                             "pickelhaube\n" +
                             "picket fence, paling\n" +
                             "pickup, pickup truck\n" +
                             "pier\n" +
                             "piggy bank, penny bank\n" +
                             "pill bottle\n" +
                             "pillow\n" +
                             "ping-pong ball\n" +
                             "pinwheel\n" +
                             "pirate, pirate ship\n" +
                             "pitcher, ewer\n" +
                             "plane, carpenter's plane, woodworking plane\n" +
                             "planetarium\n" +
                             "plastic bag\n" +
                             "plate rack\n" +
                             "plow, plough\n" +
                             "plunger, plumber's helper\n" +
                             "Polaroid camera, Polaroid Land camera\n" +
                             "pole\n" +
                             "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria\n" +
                             "poncho\n" +
                             "pool table, billiard table, snooker table\n" +
                             "pop bottle, soda bottle\n" +
                             "pot, flowerpot\n" +
                             "potter's wheel\n" +
                             "power drill\n" +
                             "prayer rug, prayer mat\n" +
                             "printer\n" +
                             "prison, prison house\n" +
                             "projectile, missile\n" +
                             "projector\n" +
                             "puck, hockey puck\n" +
                             "punching bag, punch bag, punching ball, punchball\n" +
                             "purse\n" +
                             "quill, quill pen\n" +
                             "quilt, comforter, comfort, puff\n" +
                             "racer, race car, racing car\n" +
                             "racket, racquet\n" +
                             "radiator\n" +
                             "radio, wireless\n" +
                             "radio telescope, radio reflector\n" +
                             "rain barrel\n" +
                             "recreational vehicle, RV, R.V.\n" +
                             "reel\n" +
                             "reflex camera\n" +
                             "refrigerator, icebox\n" +
                             "remote control, remote\n" +
                             "restaurant, eating house, eating place, eatery\n" +
                             "revolver, six-gun, six-shooter\n" +
                             "rifle\n" +
                             "rocking chair, rocker\n" +
                             "rotisserie\n" +
                             "rubber eraser, rubber, pencil eraser\n" +
                             "rugby ball\n" +
                             "rule, ruler\n" +
                             "running shoe\n" +
                             "safe\n" +
                             "safety pin\n" +
                             "saltshaker, salt shaker\n" +
                             "sandal\n" +
                             "sarong\n" +
                             "sax, saxophone\n" +
                             "scabbard\n" +
                             "scale, weighing machine\n" +
                             "school bus\n" +
                             "schooner\n" +
                             "scoreboard\n" +
                             "screen, CRT screen\n" +
                             "screw\n" +
                             "screwdriver\n" +
                             "seat belt, seatbelt\n" +
                             "sewing machine\n" +
                             "shield, buckler\n" +
                             "shoe shop, shoe-shop, shoe store\n" +
                             "shoji\n" +
                             "shopping basket\n" +
                             "shopping cart\n" +
                             "shovel\n" +
                             "shower cap\n" +
                             "shower curtain\n" +
                             "ski\n" +
                             "ski mask\n" +
                             "sleeping bag\n" +
                             "slide rule, slipstick\n" +
                             "sliding door\n" +
                             "slot, one-armed bandit\n" +
                             "snorkel\n" +
                             "snowmobile\n" +
                             "snowplow, snowplough\n" +
                             "soap dispenser\n" +
                             "soccer ball\n" +
                             "sock\n" +
                             "solar dish, solar collector, solar furnace\n" +
                             "sombrero\n" +
                             "soup bowl\n" +
                             "space bar\n" +
                             "space heater\n" +
                             "space shuttle\n" +
                             "spatula\n" +
                             "speedboat\n" +
                             "spider web, spider's web\n" +
                             "spindle\n" +
                             "sports car, sport car\n" +
                             "spotlight, spot\n" +
                             "stage\n" +
                             "steam locomotive\n" +
                             "steel arch bridge\n" +
                             "steel drum\n" +
                             "stethoscope\n" +
                             "stole\n" +
                             "stone wall\n" +
                             "stopwatch, stop watch\n" +
                             "stove\n" +
                             "strainer\n" +
                             "streetcar, tram, tramcar, trolley, trolley car\n" +
                             "stretcher\n" +
                             "studio couch, day bed\n" +
                             "stupa, tope\n" +
                             "submarine, pigboat, sub, U-boat\n" +
                             "suit, suit of clothes\n" +
                             "sundial\n" +
                             "sunglass\n" +
                             "sunglasses, dark glasses, shades\n" +
                             "sunscreen, sunblock, sun blocker\n" +
                             "suspension bridge\n" +
                             "swab, swob, mop\n" +
                             "sweatshirt\n" +
                             "swimming trunks, bathing trunks\n" +
                             "swing\n" +
                             "switch, electric switch, electrical switch\n" +
                             "syringe\n" +
                             "table lamp\n" +
                             "tank, army tank, armored combat vehicle, armoured combat vehicle\n" +
                             "tape player\n" +
                             "teapot\n" +
                             "teddy, teddy bear\n" +
                             "television, television system\n" +
                             "tennis ball\n" +
                             "thatch, thatched roof\n" +
                             "theater curtain, theatre curtain\n" +
                             "thimble\n" +
                             "thresher, thrasher, threshing machine\n" +
                             "throne\n" +
                             "tile roof\n" +
                             "toaster\n" +
                             "tobacco shop, tobacconist shop, tobacconist\n" +
                             "toilet seat\n" +
                             "torch\n" +
                             "totem pole\n" +
                             "tow truck, tow car, wrecker\n" +
                             "toyshop\n" +
                             "tractor\n" +
                             "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi\n" +
                             "tray\n" +
                             "trench coat\n" +
                             "tricycle, trike, velocipede\n" +
                             "trimaran\n" +
                             "tripod\n" +
                             "triumphal arch\n" +
                             "trolleybus, trolley coach, trackless trolley\n" +
                             "trombone\n" +
                             "tub, vat\n" +
                             "turnstile\n" +
                             "typewriter keyboard\n" +
                             "umbrella\n" +
                             "unicycle, monocycle\n" +
                             "upright, upright piano\n" +
                             "vacuum, vacuum cleaner\n" +
                             "vase\n" +
                             "vault\n" +
                             "velvet\n" +
                             "vending machine\n" +
                             "vestment\n" +
                             "viaduct\n" +
                             "violin, fiddle\n" +
                             "volleyball\n" +
                             "waffle iron\n" +
                             "wall clock\n" +
                             "wallet, billfold, notecase, pocketbook\n" +
                             "wardrobe, closet, press\n" +
                             "warplane, military plane\n" +
                             "washbasin, handbasin, washbowl, lavabo, wash-hand basin\n" +
                             "washer, automatic washer, washing machine\n" +
                             "water bottle\n" +
                             "water jug\n" +
                             "water tower\n" +
                             "whiskey jug\n" +
                             "whistle\n" +
                             "wig\n" +
                             "window screen\n" +
                             "window shade\n" +
                             "Windsor tie\n" +
                             "wine bottle\n" +
                             "wing\n" +
                             "wok\n" +
                             "wooden spoon\n" +
                             "wool, woolen, woollen\n" +
                             "worm fence, snake fence, snake-rail fence, Virginia fence\n" +
                             "wreck\n" +
                             "yawl\n" +
                             "yurt\n" +
                             "web site, website, internet site, site\n" +
                             "comic book\n" +
                             "crossword puzzle, crossword\n" +
                             "street sign\n" +
                             "traffic light, traffic signal, stoplight\n" +
                             "book jacket, dust cover, dust jacket, dust wrapper\n" +
                             "menu\n" +
                             "plate\n" +
                             "guacamole\n" +
                             "consomme\n" +
                             "hot pot, hotpot\n" +
                             "trifle\n" +
                             "ice cream, icecream\n" +
                             "ice lolly, lolly, lollipop, popsicle\n" +
                             "French loaf\n" +
                             "bagel, beigel\n" +
                             "pretzel\n" +
                             "cheeseburger\n" +
                             "hotdog, hot dog, red hot\n" +
                             "mashed potato\n" +
                             "head cabbage\n" +
                             "broccoli\n" +
                             "cauliflower\n" +
                             "zucchini, courgette\n" +
                             "spaghetti squash\n" +
                             "acorn squash\n" +
                             "butternut squash\n" +
                             "cucumber, cuke\n" +
                             "artichoke, globe artichoke\n" +
                             "bell pepper\n" +
                             "cardoon\n" +
                             "mushroom\n" +
                             "Granny Smith\n" +
                             "strawberry\n" +
                             "orange\n" +
                             "lemon\n" +
                             "fig\n" +
                             "pineapple, ananas\n" +
                             "banana\n" +
                             "jackfruit, jak, jack\n" +
                             "custard apple\n" +
                             "pomegranate\n" +
                             "hay\n" +
                             "carbonara\n" +
                             "chocolate sauce, chocolate syrup\n" +
                             "dough\n" +
                             "meat loaf, meatloaf\n" +
                             "pizza, pizza pie\n" +
                             "potpie\n" +
                             "burrito\n" +
                             "red wine\n" +
                             "espresso\n" +
                             "cup\n" +
                             "eggnog\n" +
                             "alp\n" +
                             "bubble\n" +
                             "cliff, drop, drop-off\n" +
                             "coral reef\n" +
                             "geyser\n" +
                             "lakeside, lakeshore\n" +
                             "promontory, headland, head, foreland\n" +
                             "sandbar, sand bar\n" +
                             "seashore, coast, seacoast, sea-coast\n" +
                             "valley, vale\n" +
                             "volcano\n" +
                             "ballplayer, baseball player\n" +
                             "groom, bridegroom\n" +
                             "scuba diver\n" +
                             "rapeseed\n" +
                             "daisy\n" +
                             "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum\n" +
                             "corn\n" +
                             "acorn\n" +
                             "hip, rose hip, rosehip\n" +
                             "buckeye, horse chestnut, conker\n" +
                             "coral fungus\n" +
                             "agaric\n" +
                             "gyromitra\n" +
                             "stinkhorn, carrion fungus\n" +
                             "earthstar\n" +
                             "hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa\n" +
                             "bolete\n" +
                             "ear, spike, capitulum\n" +
                             "toilet tissue, toilet paper, bathroom tissue\n").split("\n")).map(x -> x.trim()).collect(Collectors.toList());
  }
  
  /**
   * The type Vgg 16 zip.
   */
  static class VGG16_Zip extends VGG16 {
    
    private final NNLayer network;
    
    private VGG16_Zip(NNLayer network) {this.network = network;}
    
    /**
     * Gets network.
     *
     * @return the network
     */
    public NNLayer getNetwork() {
      return this.network;
    }
    
  }
  
}
