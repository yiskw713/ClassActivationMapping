% ==============================================================================
% UMD Part Afforance Dataset
% Version 1.0
% ------------------------------------------------------------------------------
% If you use the dataset please cite:
%
% Austin Myers, Ching L. Teo, Cornelia Fermuller, Yiannis Aloimonos.
% Affordance Detection of Tool Parts from Geometric Features.
% International Conference on Robotics and Automation (ICRA). 2015.
%
% Any bugs or questions, please email amyers AT umiacs DOT umd.DOT edu.
% ==============================================================================

This dataset contains RGB-D images for 105 kitchen, workshop, and gardening tools
from different viewpoints as well as 3 cluttered scenes. The data is in the
"tools/" and "clutter/" subdirectories respectively. Data is further organized
by each tool or scene. Finally, there are 4 types of files for each image:

  1. rgb image (*_rgb.jpg)
     The color portion of an RGB-D frame, 480x640x3.

  2. depth image (*_depth.png)
     The depth portion of an RGB-D frame, 480x640x1.

  3. label image (*_label.mat)
     Where the variable gt_label is 480x640x1
     and each pixel is labeled with its most likely affordance,
     where the integer at gt_label(i,j) indicates the affordance
     of pixel (i,j) in the image. Background pixels have zero values.

  4. ranked label image (*_label_rank.mat)
     Where the variable gt_label is 480x640x7
     Each pixel is labeled with rankings for the 7 different affordances,
     where the integer at gt_label(i,j,k) indicates the rank of the kth
     affordance. A value of 1 is the highest ranked (most likely) label,
     and we allow for ties in the ranking. Background pixels are not labeled,
     and have all zero values.

The ground truth labels are:

  1 - 'grasp'
  2 - 'cut'
  3 - 'scoop'
  4 - 'contain'
  5 - 'pound'
  6 - 'support'
  7 - 'wrap-grasp'

Ground truth labels are available for every third image in each directory,
and automatically generated labels are available for the remaining frames.
The automatically labelled frames were not used for training or testing
in our experiments.

The dataset also has several text files detailing splits for the dataset.
Text files lists each tool name (corresponding to its subdirectory) with its
corresponding split information.

- category_split.txt
    2-way split of the tools in the dataset.
- category_10_fold.txt
    10-fold leave-one-out for tools in the dataset,
    where each column corresponds to a fold,
    and 1 indicates if the tool is left out for testing.
- tool_categories.txt
    Lists the how each tool belongs to one of 17 groups.
- novel_split.txt
    Splits the data so that the testing set contains tools
    from "novel" categories not seen during training.

% ==============================================================================
% Tool Dataset Details
% ==============================================================================

% -------------------------------|
%    Dataset Objects Overview    |
% -------------------------------|
% Cut ----------------------  25 |
%   knife               12       |
%   saw                 3        |
%   scissors            8        |
%   shears              2        |
% Scoop --------------------  17 |
%   scoop               2        |
%   spoon               10       |
%   trowel              5        |
% Contain ------------------  43 |
%   bowl                10       |
%   cup                 6        |
%   ladle               5        |
%   mug                 20       |
%   pot                 2        |
% Support ------------------  10 |
%   shovel              2        |
%   turner              8        |
% Pound --------------------  10 |
%   hammer              4        |
%   mallet              4        |
%   tenderizer          2        |
% -------------------------------|
% Total -------------------- 105 |
% -------------------------------|

% ==============================================================================
% Cutting Tools
% ==============================================================================
% ---- knife ----
knife_01        cornelia's bread knife (wusthof dreizack)
knife_02        austin's bread knife
knife_03        cornelia's kitchen knife (wusthof dreizack)
knife_04        kitchen knife (cutco)
knife_05        kitchen knife (cook at home)
knife_06        blue plastic kitchen knife (seasung industrial co.ltd. (Target))
knife_07        austin's kitchen knife (JA Henckels brand)
knife_08        paring knife
knife_09        deba knife (ikea)
knife_10        knife (chefmate)
knife_11        knife (wiltshire staysharp)
knife_12        carving knife (J.A. Bornschaft)
% ---- saw ----
saw_01          hard tooth saw (Stanley)
saw_02          fine tooth utility saw (Stanley)
saw_03          mitre back saw (Buck Bros)
% ---- scissors ----
scissors_01     June's scissors
scissors_02     trojka blue kitchen scissors (ikea)
scissors_03     white scissors
scissors_04     scissors (workforce)
scissors_05     thin handle scissors (office max)
scissors_06     ergonomic handle scissors (office max)
scissors_07     kitchen scissors (wusthof dreizack)
scissors_08     scissors (dahle)
% ---- shears ----
shears_01       garden pruning shears (Fiskars)
shears_02       ergonomic angled pruning head garden shears (Fiskars)

% ==============================================================================
% Scooping Tools
% ==============================================================================
% ---- trowel ----
trowel_01       garden trowel (Ames true temper from Home Depot)
trowel_02       orange tip plastic garden trowel (Fiskars)
trowel_03       yellow tip plastic garden trowel (Fiskars)
trowel_04       blue metal small hand trowel
trowel_05       blue metal small hand trowel
% ---- spoon ---- (in order of convexity)
spoon_01        wooden asymmetric soup spoon
spoon_02        gray plastic spoon with teal blue handle
spoon_03        long wooden soup spoon
spoon_04        red soup spoon with metallic handle (giada)
spoon_05        metal soup spoon (ikea)
spoon_06        wooden soup spoon (kitchenaid)
spoon_07        wooden soup spoon
spoon_08        bamboo soup spoon (chefmate)
spoon_09        wooden soup stirring spoon (everyday living)
spoon_10        wooden soup stirring spoon
% ---- scoop ----
scoop_01        black plastic food scoop
scoop_02        blue plastic 1 cup food scoop

% ==============================================================================
% Containers
% ==============================================================================
% ---- ladle ----
ladle_01        metal ladle black rubber handle (OXO)
ladle_02        metal ladle black plastic handle (Good Cook)
ladle_03        plastic ladle with yellow silicone handle
ladle_04        red plastic ladle (chefmate)
ladle_05        wooden ladle
% ---- cup ----
cup_01          paper cup with colored line design (sysco)
cup_02          styrofoam cup
cup_03          blue plastic cup (genpack)
cup_04          multicolor paper cup (dixie)
cup_05          paper coffee cup with soft texture and coffee cup logo (dixie)
cup_06          white paper cup with green and black logo (starbucks)
% ---- mug ----
mug_01          off-white / gray mug
mug_02          las vegas themed mug
mug_03          black mug with blue inside
mug_04          UMD Alumni mug
mug_05          light denim blue mug
mug_06          royal blue quaker mug
mug_07          eggshell light green mug with unusual handle
mug_08          marigold yellow orange mug
mug_09          white mug with "Neal's Yard" tree decal
mug_10          white taper mug with ring finger handle
mug_11          white ICCV mug
mug_12          white mug with flower design
mug_13          white mug with red sailing ship (lakeforest dental associates)
mug_14          beige mug with black text
mug_15          columbia university mug
mug_16          NASA mug
mug_17          plastic mug with lip and red text
mug_18          white nespresso mug
mug_19          UMD NACS (neuro cognitive science) mug
mug_20          bada bean coffee themed mug
% ---- bowl ----
bowl_01         small hemispherical teal bowl (summer brand from Target)
bowl_02         white inside teal outside cereal bowl (Target brand)
bowl_03         two color bowl (the cellar)
bowl_04         white cereal bowl (ikea)
bowl_05         brown rice bowl black inside
bowl_06         large brown rice bowl black inside
bowl_07         large black rice bowl red inside
bowl_08         rice bowl with blue rabbit design
bowl_09         shallow bowl with red inside and black outside
bowl_10         black rice bowl with red inside
% ---- pot ----
pot_01          teal flower pot (4 in electric flower pot Home Depot)
pot_02          red clay flower pot (4.25 in Home Depot)

% ==============================================================================
% Support Tools
% ==============================================================================
% ---- turner ----
turner_01       wooden turner (OXO)
turner_02       wooden turner with slots (chefmate)
turner_03       softworks turner (OXO)
turner_04       red turner with holes (Giada)
turner_05       off white plastic turner with one slot
turner_06       red turner with blue handle and slots
turner_08       black plastic turner with slots (OXO)
turner_08       metallic turner with slots (IKEA)
% ---- shovel ----
shovel_01       green metal small hand shovel
shovel_02       light green metal small hand shovel

% ==============================================================================
% Pounding Tools
% ==============================================================================
% ---- tenderizer ----
tenderizer_01   small block head tenderizer
tenderizer_02   large hour glass head tenderizer
% ---- mallet ----
mallet_01       wooden crab mallet with cylindrical head
mallet_02       wooden crab mallet with block head
mallet_03       rubber mallet
mallet_04       rubber mallet (estwing deadhead mallet)
% ---- hammer ----
hammer_01       sledge hammer
hammer_02       claw hammer (stanley light duty)
hammer_03       claw hammer
hammer_04       claw hammer (workforce, hickory)
