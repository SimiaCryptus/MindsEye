First, define a model:

This is a very simple model that performs basic logistic regression. It is expected to be trainable to about 91% accuracy on MNIST.

Code from [MNistDemo.java:137](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L137) executed in 0.01 seconds: 
```java
    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(28,28,1));
    network.add(new DenseSynapseLayer(new int[]{28,28,1},new int[]{10})
      .setWeights(()->0.001*(Math.random()-0.45)));
    network.add(new SoftmaxActivationLayer());
    return network;
```

Returns: 

```
    PipelineNetwork/b972ebcf-164e-4958-9425-0b8e00000007
```



We use the standard MNIST dataset, made available by a helper function. In order to use data, we convert it into data tensors; helper functions are defined to work with images.

Code from [MNistDemo.java:120](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L120) executed in 0.28 seconds: 
```java
    try {
      return MNIST.trainingDataStream().map(labeledObject -> {
        Tensor categoryTensor = new Tensor(10);
        int category = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
        categoryTensor.set(category, 1);
        return new Tensor[]{labeledObject.data, categoryTensor};
      }).toArray(i->new Tensor[i][]);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
```

Returns: 

```
    [[Lcom.simiacryptus.util.ml.Tensor;@524a076e
```



Code from [MNistDemo.java:169](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L169) executed in 300.40 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = ScheduledSampleTrainable.Pow(trainingData, supervisedNetwork, 1000, 1, 0.0);
    L12Normalizer normalizer = new L12Normalizer(trainable) {
      @Override
      protected double getL1(NNLayer layer) {
        if(layer instanceof DenseSynapseLayer) return 0.001;
        return 0;
      }
  
      @Override
      protected double getL2(NNLayer layer) {
        return 0;
      }
    };
    IterativeTrainer trainer = new IterativeTrainer(normalizer);
    trainer.setScaling(new ArmijoWolfeConditions().setC1(1e-4).setC2(0.9));
    trainer.setOrientation(new TrustRegionStrategy(new LBFGS().setMinHistory(5)) {
      @Override
      public TrustRegion getRegionPolicy(NNLayer layer) {
        if(layer instanceof DenseSynapseLayer) return new SingleOrthant();
        return null;
      }
    });
    trainer.setMonitor(new TrainingMonitor(){
      @Override
      public void log(String msg) {
        System.out.print(msg);
      }
  
      @Override
      public void onStepComplete(IterativeTrainer.Step currentPoint) {
        super.onStepComplete(currentPoint);
      }
    });
    trainer.setTimeout(5, TimeUnit.MINUTES).run();
```
Logging: 
```
    ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 1.2457309396155174 - Infinity	th(alpha)=13550.376887 > -12819915.944598;th'(alpha)=0.000000 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 0.6228654698077587 - 1.2457309396155174	th(alpha)=6782.943765 > -6409956.613088;th'(alpha)=0.000000 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 0.31143273490387935 - 0.6228654698077587	th(alpha)=3399.227205 > -3204976.947333;th'(alpha)=0.000000 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 0.15571636745193967 - 0.31143273490387935	th(alpha)=1707.368924 > -1602487.114456;th'(alpha)=0.000000 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 0.07785818372596984 - 0.15571636745193967	th(alpha)=861.439784 > -801242.198017;th'(alpha)=0.000000 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 0.03892909186298492 - 0.07785818372596984	th(alpha)=438.475214 > -400619.739798;th'(alpha)=0.000000 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 0.01946454593149246 - 0.03892909186298492	th(alpha)=226.992929 > -200308.510688;th'(alpha)=0.000000 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 0.00973227296574623 - 0.01946454593149246	th(alpha)=121.251786 > -100152.896133;th'(alpha)=0.000000 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 0.004866136482873115 - 0.00973227296574623	th(alpha)=68.381215 > -50075.088856;th'(alpha)=0.000000 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 0.0024330682414365574 - 0.004866136482873115	th(alpha)=41.945929 > -25036.185217;th'(alpha)=0.000000 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 0.0012165341207182787 - 0.0024330682414365574	th(alpha)=28.728286 > -12516.733398;th'(alpha)=0.000000 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 6.082670603591394E-4 - 0.0012165341207182787	th(alpha)=22.119465 > -6257.007488;th'(alpha)=0.510258 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 3.041335301795697E-4 - 6.082670603591394E-4	th(alpha)=18.802522 > -3127.144533;th'(alpha)=14.656722 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 1.5206676508978484E-4 - 3.041335301795697E-4	th(alpha)=17.142019 > -1562.213056;th'(alpha)=1361.991670 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 7.603338254489242E-5 - 1.5206676508978484E-4	th(alpha)=16.311842 > -779.747317;th'(alpha)=2061.113014 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 3.801669127244621E-5 - 7.603338254489242E-5	th(alpha)=15.892794 > -388.514448;th'(alpha)=24.323380 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 1.9008345636223105E-5 - 3.801669127244621E-5	th(alpha)=15.668288 > -192.898013;th'(alpha)=1.817943 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 9.504172818111552E-6 - 1.9008345636223105E-5	th(alpha)=15.510755 > -95.089796;th'(alpha)=174.164435 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 4.752086409055776E-6 - 9.504172818111552E-6	th(alpha)=15.382783 > -46.185687;th'(alpha)=97.189009 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 2.376043204527888E-6 - 4.752086409055776E-6	th(alpha)=15.265819 > -21.733633;th'(alpha)=241.448315 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 1.188021602263944E-6 - 2.376043204527888E-6	th(alpha)=15.018774 > -9.507605;th'(alpha)=606.673481 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 5.94010801131972E-7 - 1.188021602263944E-6	th(alpha)=14.264316 > -3.394592;th'(alpha)=1830.892039 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 2.97005400565986E-7 - 5.94010801131972E-7	th(alpha)=11.817779 > -0.338085;th'(alpha)=5725.573380 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 1.48502700282993E-7 - 2.97005400565986E-7	th(alpha)=7.535272 > 1.190168;th'(alpha)=6061.234407 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 7.42513501414965E-8 - 1.48502700282993E-7	th(alpha)=4.095279 > 1.954295;th'(alpha)=3337.373387 >= -92619733762.725660ARMIJO: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 3.712567507074825E-8 - 7.42513501414965E-8	th(alpha)=2.406706 > 2.336358;th'(alpha)=1421.479775 >= -92619733762.725660END: th(0)=2.718422;th'(0)=-102910815291.917400;	0.0 - 1.8562837535374126E-8 - 3.712567507074825E-8	th(alpha)=1.893834;th'(alpha)=124.331630Iteration 1 complete. Error: 1.8938342170559799ARMIJO: th(0)=1.892951;th'(0)=-155500663091.961760;	0.0 - 2.3124301044871804E-8 - Infinity	th(alpha)=2.434428 > 1.533366;th'(alpha)=1927.776972 >= -139950596782.765600END: th(0)=1.892951;th'(0)=-155500663091.961760;	0.0 - 1.1562150522435902E-8 - 2.3124301044871804E-8	th(alpha)=1.544838;th'(alpha)=535.807934Iteration 2 complete. Error: 1.544838451012342END: th(0)=1.546184;th'(0)=-94179285092.695130;	0.0 - 1.4403328634290123E-8 - Infinity	th(alpha)=1.266591;th'(alpha)=599.385095Iteration 3 complete. Error: 1.2665914142147638ARMIJO: th(0)=1.268033;th'(0)=-61891799394.973080;	0.0 - 1.7942672113185322E-8 - Infinity	th(alpha)=1.174309 > 1.156982;th'(alpha)=925.847925 >= -55702619455.475780END: th(0)=1.268033;th'(0)=-61891799394.973080;	0.0 - 8.971336056592661E-9 - 1.7942672113185322E-8	th(alpha)=0.958421;th'(alpha)=-60.235209Iteration 4 complete. Error: 0.9584207663952887END: th(0)=0.958848;th'(0)=-15190618980.932812;	0.0 - 1.1175870895385745E-8 - Infinity	th(alpha)=0.878168;th'(alpha)=-16.615371Iteration 5 complete. Error: 0.8781680736999875WOLFE: th(0)=0.879261;th'(0)=-230.400319;	0.0 - 1.3922128151530598E-8 - Infinity	th(alpha)=0.879261 <= 0.879261;th'(alpha)=-236.865858 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	1.3922128151530598E-8 - 2.7844256303061195E-8 - Infinity	th(alpha)=0.879261 <= 0.879261;th'(alpha)=-228.869968 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	2.7844256303061195E-8 - 5.568851260612239E-8 - Infinity	th(alpha)=0.879261 <= 0.879261;th'(alpha)=-234.071455 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	5.568851260612239E-8 - 1.1137702521224478E-7 - Infinity	th(alpha)=0.879261 <= 0.879261;th'(alpha)=-230.031156 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	1.1137702521224478E-7 - 2.2275405042448956E-7 - Infinity	th(alpha)=0.879261 <= 0.879261;th'(alpha)=-233.742598 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	2.2275405042448956E-7 - 4.455081008489791E-7 - Infinity	th(alpha)=0.879261 <= 0.879261;th'(alpha)=-228.400939 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	4.455081008489791E-7 - 8.910162016979582E-7 - Infinity	th(alpha)=0.879261 <= 0.879261;th'(alpha)=-238.354533 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	8.910162016979582E-7 - 1.7820324033959165E-6 - Infinity	th(alpha)=0.879260 <= 0.879261;th'(alpha)=-232.919990 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	1.7820324033959165E-6 - 3.564064806791833E-6 - Infinity	th(alpha)=0.879260 <= 0.879261;th'(alpha)=-225.365406 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	3.564064806791833E-6 - 7.128129613583666E-6 - Infinity	th(alpha)=0.879259 <= 0.879261;th'(alpha)=-238.351135 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	7.128129613583666E-6 - 1.4256259227167332E-5 - Infinity	th(alpha)=0.879257 <= 0.879260;th'(alpha)=-225.800860 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	1.4256259227167332E-5 - 2.8512518454334664E-5 - Infinity	th(alpha)=0.879254 <= 0.879260;th'(alpha)=-227.226443 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	2.8512518454334664E-5 - 5.702503690866933E-5 - Infinity	th(alpha)=0.879247 <= 0.879260;th'(alpha)=-234.190431 < -207.360287WOLFE: th(0)=0.879261;th'(0)=-230.400319;	5.702503690866933E-5 - 1.1405007381733866E-4 - Infinity	th(alpha)=0.879233 <= 0.879258;th'(alpha)=-230.583204 < -207.360287WOL
    ...skipping 274837 bytes...
    ha)=0.118226;th'(alpha)=0.011185Iteration 492 complete. Error: 0.11822564053340028END: th(0)=0.118162;th'(0)=-2.255464;	0.0 - 0.01248246146442363 - Infinity	th(alpha)=0.118143;th'(alpha)=-2.162634Iteration 493 complete. Error: 0.11814343599588423END: th(0)=0.118080;th'(0)=-0.821133;	0.0 - 0.015549788448790935 - Infinity	th(alpha)=0.118071;th'(alpha)=-0.012488Iteration 494 complete. Error: 0.11807133162388227END: th(0)=0.118008;th'(0)=-25.087636;	0.0 - 0.01937085257513485 - Infinity	th(alpha)=0.117745;th'(alpha)=-0.308254Iteration 495 complete. Error: 0.11774482158813945END: th(0)=0.117682;th'(0)=-2.393430;	0.0 - 0.0241308703795764 - Infinity	th(alpha)=0.117647;th'(alpha)=-2.127397Iteration 496 complete. Error: 0.11764700961332364END: th(0)=0.117584;th'(0)=-1.726680;	0.0 - 0.030060571831689965 - Infinity	th(alpha)=0.117544;th'(alpha)=-0.059413Iteration 497 complete. Error: 0.11754420224251554END: th(0)=0.117481;th'(0)=-1.004643;	0.0 - 0.037447384393270895 - Infinity	th(alpha)=0.117457;th'(alpha)=-0.036693Iteration 498 complete. Error: 0.11745717775875446END: th(0)=0.117394;th'(0)=-2.379811;	0.0 - 0.04664936534637282 - Infinity	th(alpha)=0.117326;th'(alpha)=-2.169813Iteration 499 complete. Error: 0.11732615359749787END: th(0)=0.125485;th'(0)=-10.553128;	0.0 - 0.058112557725404565 - Infinity	th(alpha)=0.125179;th'(alpha)=-5.945518Iteration 500 complete. Error: 0.12517947709346877END: th(0)=0.125112;th'(0)=-11.767781;	0.0 - 0.07239261113872922 - Infinity	th(alpha)=0.124637;th'(alpha)=-0.574340Iteration 501 complete. Error: 0.12463697568325055END: th(0)=0.124570;th'(0)=-8.578941;	0.0 - 0.09018171549506993 - Infinity	th(alpha)=0.124114;th'(alpha)=-0.550836Iteration 502 complete. Error: 0.12411421746305674END: th(0)=0.124047;th'(0)=-1.698297;	0.0 - 0.11234215317981272 - Infinity	th(alpha)=0.123929;th'(alpha)=-1.371328Iteration 503 complete. Error: 0.12392866051575534END: th(0)=0.123862;th'(0)=-1.327511;	0.0 - 0.13994809603911848 - Infinity	th(alpha)=0.123738;th'(alpha)=-0.171057Iteration 504 complete. Error: 0.12373797772346866END: th(0)=0.123671;th'(0)=-1.290833;	0.0 - 0.17433767317621374 - Infinity	th(alpha)=0.123533;th'(alpha)=-1.076873Iteration 505 complete. Error: 0.12353335843702368END: th(0)=0.123467;th'(0)=-1.334090;	0.0 - 0.2171778334161877 - Infinity	th(alpha)=0.123283;th'(alpha)=-0.220808Iteration 506 complete. Error: 0.12328320048598673END: th(0)=0.123217;th'(0)=-0.510626;	0.0 - 0.27054514648520983 - Infinity	th(alpha)=0.123130;th'(alpha)=-0.115004Iteration 507 complete. Error: 0.12312994824695657END: th(0)=0.123094;th'(0)=-0.879850;	0.0 - 0.3370264595394382 - Infinity	th(alpha)=0.122949;th'(alpha)=-0.187109Iteration 508 complete. Error: 0.1229485304703669END: th(0)=0.122882;th'(0)=-8.585496;	0.0 - 0.4198442881173555 - Infinity	th(alpha)=0.120459;th'(alpha)=-3.547143Iteration 509 complete. Error: 0.12045894507885911ARMIJO: th(0)=0.120395;th'(0)=-135.928369;	0.0 - 0.5230130195286413 - Infinity	th(alpha)=0.479228 > 0.120110;th'(alpha)=1357.560584 >= -134.645527ARMIJO: th(0)=0.120395;th'(0)=-135.928369;	0.0 - 0.26150650976432066 - 0.5230130195286413	th(alpha)=0.165688 > 0.120252;th'(alpha)=224.068298 >= -134.645527ARMIJO: th(0)=0.120395;th'(0)=-135.928369;	0.0 - 0.13075325488216033 - 0.26150650976432066	th(alpha)=0.121616 > 0.120324;th'(alpha)=34.782732 >= -134.645527END: th(0)=0.120395;th'(0)=-135.928369;	0.0 - 0.06537662744108017 - 0.13075325488216033	th(alpha)=0.115688;th'(alpha)=-0.792130Iteration 510 complete. Error: 0.115688158664901END: th(0)=0.133889;th'(0)=-10.508295;	0.0 - 0.08144168753107041 - Infinity	th(alpha)=0.133578;th'(alpha)=-0.200046Iteration 511 complete. Error: 0.13357844232219368END: th(0)=0.133506;th'(0)=-0.606726;	0.0 - 0.10145442993195371 - Infinity	th(alpha)=0.133467;th'(alpha)=-0.545053Iteration 512 complete. Error: 0.13346703855291567END: th(0)=0.134215;th'(0)=-0.994949;	0.0 - 0.12638492232728937 - Infinity	th(alpha)=0.134140;th'(alpha)=-0.101429Iteration 513 complete. Error: 0.13414008354747659END: th(0)=0.134067;th'(0)=-23.322219;	0.0 - 0.15744160804400836 - Infinity	th(alpha)=0.133926;th'(alpha)=2.034911Iteration 514 complete. Error: 0.13392605636599317ARMIJO: th(0)=0.134938;th'(0)=-7.764857;	0.0 - 0.19612988232324055 - Infinity	th(alpha)=0.135146 > 0.134932;th'(alpha)=2.766167 >= -7.691575END: th(0)=0.134938;th'(0)=-7.764857;	0.0 - 0.09806494116162028 - 0.19612988232324055	th(alpha)=0.134699;th'(alpha)=0.066941Iteration 515 complete. Error: 0.1346987306530636END: th(0)=0.134625;th'(0)=-1.967482;	0.0 - 0.12216253129660565 - Infinity	th(alpha)=0.134477;th'(alpha)=-1.771484Iteration 516 complete. Error: 0.13447711528720022END: th(0)=0.152528;th'(0)=-14.898185;	0.0 - 0.1521816448979306 - Infinity	th(alpha)=0.151594;th'(alpha)=-0.631048Iteration 517 complete. Error: 0.15159445520116924END: th(0)=0.152639;th'(0)=-4.408302;	0.0 - 0.1895773834909341 - Infinity	th(alpha)=0.152167;th'(alpha)=-0.635360Iteration 518 complete. Error: 0.15216695371711073END: th(0)=0.155314;th'(0)=-2.461918;	0.0 - 0.2361624120660126 - Infinity	th(alpha)=0.154969;th'(alpha)=-0.482020Iteration 519 complete. Error: 0.15496896802832716END: th(0)=0.154882;th'(0)=-9.735514;	0.0 - 0.29419482348486087 - Infinity	th(alpha)=0.153549;th'(alpha)=-1.359119Iteration 520 complete. Error: 0.15354880909696714END: th(0)=0.153465;th'(0)=-5.059611;	0.0 - 0.366487593889817 - Infinity	th(alpha)=0.152411;th'(alpha)=-1.361056Iteration 521 complete. Error: 0.15241097064140413END: th(0)=0.152326;th'(0)=-11.781151;	0.0 - 0.45654493469379187 - Infinity	th(alpha)=0.149469;th'(alpha)=-3.419057Iteration 522 complete. Error: 0.14946934660609879END: th(0)=0.149387;th'(0)=-13.907421;	0.0 - 0.5687321504728023 - Infinity	th(alpha)=0.147949;th'(alpha)=6.014281Iteration 523 complete. Error: 0.14794921322054766END: th(0)=0.147868;th'(0)=-10.834803;	0.0 - 0.7084872361980379 - Infinity	th(alpha)=0.145788;th'(alpha)=-0.559332Iteration 524 complete. Error: 0.14578782471431997END: th(0)=0.145708;th'(0)=-9.802349;	0.0 - 0.8825844704545828 - Infinity	th(alpha)=0.143942;th'(alpha)=0.801765Iteration 525 complete. Error: 0.1439421288442921END: th(0)=0.143863;th'(0)=-4.715858;	0.0 - 1.0994627816694513 - Infinity	th(alpha)=0.141296;th'(alpha)=-2.585216Iteration 526 complete. Error: 0.14129552946389579ARMIJO: th(0)=0.141218;th'(0)=-22.744240;	0.0 - 1.3696348040813762 - Infinity	th(alpha)=0.173495 > 0.141094;th'(alpha)=163.489971 >= -22.529588END: th(0)=0.141218;th'(0)=-22.744240;	0.0 - 0.6848174020406881 - 1.3696348040813762	th(alpha)=0.140415;th'(alpha)=23.796487Iteration 527 complete. Error: 0.14041546190967769END: th(0)=0.140339;th'(0)=-21.687544;	0.0 - 0.8530982257092039 - Infinity	th(alpha)=0.134324;th'(alpha)=-0.397675Iteration 528 complete. Error: 0.134323939408631END: th(0)=0.134251;th'(0)=-3.100807;	0.0 - 1.0627308542970573 - Infinity	th(alpha)=0.133172;th'(alpha)=-0.494858Iteration 529 complete. Error: 0.13317195790720454END: th(0)=0.149146;th'(0)=-4.018904;	0.0 - 1.3238767056818748 - Infinity	th(alpha)=0.146889;th'(alpha)=-1.720549Iteration 530 complete. Error: 0.1468891735280746END: th(0)=0.146809;th'(0)=-5.287698;	0.0 - 1.6491941725041777 - Infinity	th(alpha)=0.144918;th'(alpha)=6.734274Iteration 531 complete. Error: 0.14491775996484177END: th(0)=0.144838;th'(0)=-9.459829;	0.0 - 2.054452206122065 - Infinity	th(alpha)=0.139352;th'(alpha)=-4.237127Iteration 532 complete. Error: 0.13935224805313812ARMIJO: th(0)=0.139451;th'(0)=-12.042273;	0.0 - 2.5592946771276126 - Infinity	th(alpha)=0.151617 > 0.139327;th'(alpha)=125.859767 >= -11.928622END: th(0)=0.139451;th'(0)=-12.042273;	0.0 - 1.2796473385638063 - 2.5592946771276126	th(alpha)=0.132959;th'(alpha)=1.127237Iteration 533 complete. Error: 0.13295851901086828END: th(0)=0.150876;th'(0)=-14.460003;	0.0 - 1.5940962814455866 - Infinity	th(alpha)=0.142730;th'(alpha)=1.767491Iteration 534 complete. Error: 0.14272975126593898ARMIJO: th(0)=0.142652;th'(0)=-23.583039;	0.0 - 1.985815058522813 - Infinity	th(alpha)=0.158949 > 0.142465;th'(alpha)=128.300172 >= -23.360471END: th(0)=0.142652;th'(0)=-23.583039;	0.0 - 0.9929075292614065 - 1.985815058522813	th(alpha)=0.134547;th'(alpha)=1.930498Iteration 535 complete. Error: 0.13454749795786586
```

If we test our model against the entire validation dataset, we get this accuracy:

Code from [MNistDemo.java:61](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L61) executed in 0.50 seconds: 
```java
    try {
      return MNIST.validationDataStream().mapToDouble(labeledObject->{
        int actualCategory = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
        double[] predictionSignal = network.eval(labeledObject.data).data[0].getData();
        int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
        return predictionList[0]==actualCategory?1:0;
      }).average().getAsDouble() * 100;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
```

Returns: 

```
    84.69
```



Let's examine some incorrectly predicted results in more detail:

Code from [MNistDemo.java:75](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L75) executed in 0.61 seconds: 
```java
    try {
      TableOutput table = new TableOutput();
      MNIST.validationDataStream().map(labeledObject->{
        try {
          int actualCategory = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
          double[] predictionSignal = network.eval(labeledObject.data).data[0].getData();
          int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
          if(predictionList[0] == actualCategory) return null; // We will only examine mispredicted rows
          LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
          row.put("Image", log.image(labeledObject.data.toGrayImage(),labeledObject.label));
          row.put("Prediction", Arrays.stream(predictionList).limit(3)
                                    .mapToObj(i->String.format("%d (%.1f%%)",i, 100.0*predictionSignal[i]))
                                    .reduce((a,b)->a+", "+b).get());
          return row;
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }).filter(x->null!=x).limit(100).forEach(table::putRow);
      return table;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
```

Returns: 

Image | Prediction
----- | ----------
![[5]](etc/bellsAndWhistles.1.png)   | 2 (100.0%), 4 (0.0%), 6 (0.0%) 
![[6]](etc/bellsAndWhistles.2.png)   | 2 (99.9%), 6 (0.1%), 1 (0.0%)  
![[4]](etc/bellsAndWhistles.3.png)   | 6 (56.8%), 4 (43.1%), 5 (0.1%) 
![[3]](etc/bellsAndWhistles.4.png)   | 5 (62.4%), 3 (37.6%), 6 (0.0%) 
![[5]](etc/bellsAndWhistles.5.png)   | 4 (97.6%), 8 (2.0%), 5 (0.3%)  
![[6]](etc/bellsAndWhistles.6.png)   | 2 (100.0%), 6 (0.0%), 3 (0.0%) 
![[7]](etc/bellsAndWhistles.7.png)   | 2 (78.9%), 7 (21.1%), 3 (0.0%) 
![[4]](etc/bellsAndWhistles.8.png)   | 9 (76.7%), 4 (19.2%), 5 (2.6%) 
![[6]](etc/bellsAndWhistles.9.png)   | 2 (100.0%), 7 (0.0%), 6 (0.0%) 
![[7]](etc/bellsAndWhistles.10.png)  | 9 (99.5%), 5 (0.5%), 7 (0.0%)  
![[7]](etc/bellsAndWhistles.11.png)  | 3 (92.2%), 7 (7.6%), 5 (0.2%)  
![[9]](etc/bellsAndWhistles.12.png)  | 7 (66.1%), 9 (33.9%), 4 (0.0%) 
![[2]](etc/bellsAndWhistles.13.png)  | 9 (99.4%), 2 (0.6%), 8 (0.0%)  
![[4]](etc/bellsAndWhistles.14.png)  | 6 (100.0%), 4 (0.0%), 7 (0.0%) 
![[7]](etc/bellsAndWhistles.15.png)  | 4 (100.0%), 9 (0.0%), 7 (0.0%) 
![[0]](etc/bellsAndWhistles.16.png)  | 3 (98.6%), 0 (1.4%), 2 (0.0%)  
![[7]](etc/bellsAndWhistles.17.png)  | 2 (64.2%), 7 (35.8%), 9 (0.0%) 
![[2]](etc/bellsAndWhistles.18.png)  | 9 (100.0%), 8 (0.0%), 6 (0.0%) 
![[9]](etc/bellsAndWhistles.19.png)  | 2 (100.0%), 9 (0.0%), 7 (0.0%) 
![[2]](etc/bellsAndWhistles.20.png)  | 3 (100.0%), 2 (0.0%), 8 (0.0%) 
![[9]](etc/bellsAndWhistles.21.png)  | 3 (77.1%), 4 (22.9%), 9 (0.0%) 
![[6]](etc/bellsAndWhistles.22.png)  | 5 (100.0%), 0 (0.0%), 8 (0.0%) 
![[3]](etc/bellsAndWhistles.23.png)  | 5 (97.5%), 3 (2.5%), 9 (0.0%)  
![[8]](etc/bellsAndWhistles.24.png)  | 7 (100.0%), 9 (0.0%), 5 (0.0%) 
![[5]](etc/bellsAndWhistles.25.png)  | 8 (100.0%), 5 (0.0%), 9 (0.0%) 
![[9]](etc/bellsAndWhistles.26.png)  | 5 (99.2%), 3 (0.6%), 9 (0.1%)  
![[7]](etc/bellsAndWhistles.27.png)  | 3 (99.5%), 4 (0.5%), 7 (0.0%)  
![[3]](etc/bellsAndWhistles.28.png)  | 5 (100.0%), 3 (0.0%), 0 (0.0%) 
![[4]](etc/bellsAndWhistles.29.png)  | 2 (99.6%), 3 (0.3%), 6 (0.0%)  
![[8]](etc/bellsAndWhistles.30.png)  | 1 (85.2%), 8 (14.3%), 2 (0.5%) 
![[6]](etc/bellsAndWhistles.31.png)  | 0 (90.8%), 6 (9.2%), 4 (0.0%)  
![[9]](etc/bellsAndWhistles.32.png)  | 3 (99.7%), 5 (0.3%), 9 (0.0%)  
![[8]](etc/bellsAndWhistles.33.png)  | 5 (77.9%), 4 (17.7%), 0 (4.4%) 
![[8]](etc/bellsAndWhistles.34.png)  | 4 (99.9%), 1 (0.1%), 9 (0.0%)  
![[4]](etc/bellsAndWhistles.35.png)  | 6 (62.0%), 2 (36.8%), 5 (0.8%) 
![[7]](etc/bellsAndWhistles.36.png)  | 2 (100.0%), 7 (0.0%), 3 (0.0%) 
![[3]](etc/bellsAndWhistles.37.png)  | 5 (100.0%), 4 (0.0%), 3 (0.0%) 
![[2]](etc/bellsAndWhistles.38.png)  | 0 (100.0%), 2 (0.0%), 3 (0.0%) 
![[5]](etc/bellsAndWhistles.39.png)  | 4 (100.0%), 5 (0.0%), 6 (0.0%) 
![[9]](etc/bellsAndWhistles.40.png)  | 7 (100.0%), 1 (0.0%), 9 (0.0%) 
![[2]](etc/bellsAndWhistles.41.png)  | 7 (100.0%), 3 (0.0%), 2 (0.0%) 
![[4]](etc/bellsAndWhistles.42.png)  | 9 (85.5%), 4 (14.5%), 8 (0.0%) 
![[3]](etc/bellsAndWhistles.43.png)  | 2 (70.5%), 3 (29.5%), 8 (0.0%) 
![[8]](etc/bellsAndWhistles.44.png)  | 0 (100.0%), 8 (0.0%), 5 (0.0%) 
![[6]](etc/bellsAndWhistles.45.png)  | 9 (98.4%), 6 (1.4%), 4 (0.1%)  
![[3]](etc/bellsAndWhistles.46.png)  | 5 (99.5%), 3 (0.5%), 0 (0.0%)  
![[8]](etc/bellsAndWhistles.47.png)  | 5 (71.7%), 8 (28.2%), 3 (0.1%) 
![[3]](etc/bellsAndWhistles.48.png)  | 7 (89.5%), 3 (10.5%), 9 (0.0%) 
![[5]](etc/bellsAndWhistles.49.png)  | 3 (69.0%), 0 (31.0%), 5 (0.0%) 
![[1]](etc/bellsAndWhistles.50.png)  | 3 (99.5%), 1 (0.4%), 7 (0.1%)  
![[7]](etc/bellsAndWhistles.51.png)  | 3 (88.8%), 9 (11.2%), 1 (0.0%) 
![[2]](etc/bellsAndWhistles.52.png)  | 7 (100.0%), 3 (0.0%), 2 (0.0%) 
![[4]](etc/bellsAndWhistles.53.png)  | 9 (98.2%), 4 (1.8%), 3 (0.0%)  
![[6]](etc/bellsAndWhistles.54.png)  | 5 (99.6%), 9 (0.4%), 0 (0.0%)  
![[9]](etc/bellsAndWhistles.55.png)  | 7 (99.6%), 5 (0.4%), 9 (0.0%)  
![[8]](etc/bellsAndWhistles.56.png)  | 5 (99.1%), 8 (0.9%), 2 (0.0%)  
![[5]](etc/bellsAndWhistles.57.png)  | 8 (100.0%), 5 (0.0%), 3 (0.0%) 
![[8]](etc/bellsAndWhistles.58.png)  | 9 (56.7%), 2 (43.3%), 5 (0.0%) 
![[2]](etc/bellsAndWhistles.59.png)  | 0 (93.1%), 2 (6.9%), 5 (0.0%)  
![[8]](etc/bellsAndWhistles.60.png)  | 4 (98.7%), 2 (1.0%), 5 (0.3%)  
![[8]](etc/bellsAndWhistles.61.png)  | 7 (100.0%), 9 (0.0%), 8 (0.0%) 
![[8]](etc/bellsAndWhistles.62.png)  | 5 (99.4%), 8 (0.6%), 7 (0.0%)  
![[2]](etc/bellsAndWhistles.63.png)  | 8 (100.0%), 5 (0.0%), 2 (0.0%) 
![[6]](etc/bellsAndWhistles.64.png)  | 0 (100.0%), 7 (0.0%), 5 (0.0%) 
![[9]](etc/bellsAndWhistles.65.png)  | 5 (100.0%), 8 (0.0%), 9 (0.0%) 
![[3]](etc/bellsAndWhistles.66.png)  | 5 (100.0%), 3 (0.0%), 2 (0.0%) 
![[3]](etc/bellsAndWhistles.67.png)  | 5 (63.0%), 3 (37.0%), 9 (0.0%) 
![[2]](etc/bellsAndWhistles.68.png)  | 8 (100.0%), 2 (0.0%), 1 (0.0%) 
![[6]](etc/bellsAndWhistles.69.png)  | 5 (100.0%), 9 (0.0%), 6 (0.0%) 
![[3]](etc/bellsAndWhistles.70.png)  | 7 (100.0%), 3 (0.0%), 8 (0.0%) 
![[8]](etc/bellsAndWhistles.71.png)  | 9 (96.4%), 8 (3.6%), 7 (0.0%)  
![[7]](etc/bellsAndWhistles.72.png)  | 2 (94.4%), 7 (5.6%), 9 (0.0%)  
![[5]](etc/bellsAndWhistles.73.png)  | 8 (73.3%), 5 (26.7%), 9 (0.0%) 
![[5]](etc/bellsAndWhistles.74.png)  | 8 (56.8%), 2 (43.2%), 5 (0.0%) 
![[9]](etc/bellsAndWhistles.75.png)  | 3 (100.0%), 7 (0.0%), 5 (0.0%) 
![[5]](etc/bellsAndWhistles.76.png)  | 7 (100.0%), 0 (0.0%), 8 (0.0%) 
![[8]](etc/bellsAndWhistles.77.png)  | 4 (56.6%), 5 (42.3%), 8 (0.9%) 
![[9]](etc/bellsAndWhistles.78.png)  | 7 (99.0%), 9 (1.0%), 3 (0.0%)  
![[8]](etc/bellsAndWhistles.79.png)  | 0 (79.5%), 2 (11.9%), 5 (8.4%) 
![[4]](etc/bellsAndWhistles.80.png)  | 7 (97.2%), 4 (2.2%), 9 (0.6%)  
![[5]](etc/bellsAndWhistles.81.png)  | 3 (99.9%), 5 (0.1%), 1 (0.0%)  
![[3]](etc/bellsAndWhistles.82.png)  | 5 (100.0%), 3 (0.0%), 9 (0.0%) 
![[6]](etc/bellsAndWhistles.83.png)  | 5 (61.8%), 8 (36.2%), 0 (2.0%) 
![[4]](etc/bellsAndWhistles.84.png)  | 1 (94.0%), 8 (6.0%), 7 (0.0%)  
![[6]](etc/bellsAndWhistles.85.png)  | 2 (100.0%), 3 (0.0%), 7 (0.0%) 
![[3]](etc/bellsAndWhistles.86.png)  | 8 (63.7%), 2 (36.3%), 3 (0.0%) 
![[3]](etc/bellsAndWhistles.87.png)  | 5 (100.0%), 0 (0.0%), 3 (0.0%) 
![[3]](etc/bellsAndWhistles.88.png)  | 2 (100.0%), 3 (0.0%), 8 (0.0%) 
![[9]](etc/bellsAndWhistles.89.png)  | 4 (99.9%), 9 (0.1%), 2 (0.0%)  
![[3]](etc/bellsAndWhistles.90.png)  | 6 (100.0%), 3 (0.0%), 8 (0.0%) 
![[8]](etc/bellsAndWhistles.91.png)  | 3 (84.4%), 2 (15.6%), 4 (0.0%) 
![[0]](etc/bellsAndWhistles.92.png)  | 9 (100.0%), 5 (0.0%), 0 (0.0%) 
![[8]](etc/bellsAndWhistles.93.png)  | 6 (100.0%), 8 (0.0%), 5 (0.0%) 
![[9]](etc/bellsAndWhistles.94.png)  | 7 (92.9%), 9 (7.1%), 8 (0.0%)  
![[3]](etc/bellsAndWhistles.95.png)  | 5 (100.0%), 3 (0.0%), 4 (0.0%) 
![[8]](etc/bellsAndWhistles.96.png)  | 5 (100.0%), 9 (0.0%), 8 (0.0%) 
![[3]](etc/bellsAndWhistles.97.png)  | 8 (92.1%), 9 (6.6%), 3 (1.3%)  
![[8]](etc/bellsAndWhistles.98.png)  | 2 (100.0%), 8 (0.0%), 1 (0.0%) 
![[2]](etc/bellsAndWhistles.99.png)  | 3 (48.5%), 2 (31.2%), 8 (20.3%)
![[5]](etc/bellsAndWhistles.100.png) | 9 (99.3%), 5 (0.7%), 4 (0.0%)  




