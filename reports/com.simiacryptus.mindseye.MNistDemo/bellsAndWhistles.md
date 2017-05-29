First, define a model:

Code from [MNistDemo.java:126](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L126) executed in 0.01 seconds: 
```java
    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(28,28,1));
    network.add(new DenseSynapseLayer(new int[]{28,28,1},new int[]{10})
      .setWeights(()->0.001*(Math.random()-0.45)));
    network.add(new BiasLayer(10));
    network.add(new ReLuActivationLayer());
    network.add(new SoftmaxActivationLayer());
    return network;
```

Returns: 

```
    PipelineNetwork/cc7d6c7e-682d-4834-8c1f-074400000009
```



Code from [MNistDemo.java:111](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L111) executed in 0.29 seconds: 
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
    [[Lcom.simiacryptus.util.ml.Tensor;@16681017
```



Code from [MNistDemo.java:160](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L160) executed in 300.40 seconds: 
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
    trainer.setScaling(new ArmijoWolfeConditions().setC1(1e-5).setC2(0.8));
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
    ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 1.2457309396155174 - Infinity	th(alpha)=14159.582487 > -1392412.672217;th'(alpha)=0.000000 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 0.6228654698077587 - 1.2457309396155174	th(alpha)=7092.229788 > -696205.112895;th'(alpha)=0.000000 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 0.31143273490387935 - 0.6228654698077587	th(alpha)=3558.553439 > -348101.333234;th'(alpha)=0.000000 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 0.15571636745193967 - 0.31143273490387935	th(alpha)=1791.715265 > -174049.443403;th'(alpha)=0.000000 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 0.07785818372596984 - 0.15571636745193967	th(alpha)=908.296177 > -87023.498488;th'(alpha)=0.000000 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 0.03892909186298492 - 0.07785818372596984	th(alpha)=466.586634 > -43510.526030;th'(alpha)=0.000000 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 0.01946454593149246 - 0.03892909186298492	th(alpha)=245.731862 > -21754.039801;th'(alpha)=0.000000 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 0.00973227296574623 - 0.01946454593149246	th(alpha)=135.304476 > -10875.796687;th'(alpha)=0.000000 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 0.004866136482873115 - 0.00973227296574623	th(alpha)=80.090783 > -5436.675130;th'(alpha)=0.000000 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 0.0024330682414365574 - 0.004866136482873115	th(alpha)=44.026585 > -2717.114351;th'(alpha)=-0.000039 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 0.0012165341207182787 - 0.0024330682414365574	th(alpha)=30.223162 > -1357.333962;th'(alpha)=0.000083 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 6.082670603591394E-4 - 0.0012165341207182787	th(alpha)=23.321450 > -677.443767;th'(alpha)=-0.035308 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 3.041335301795697E-4 - 6.082670603591394E-4	th(alpha)=19.870596 > -337.498670;th'(alpha)=-0.005362 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 1.5206676508978484E-4 - 3.041335301795697E-4	th(alpha)=18.145182 > -167.526121;th'(alpha)=0.021277 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 7.603338254489242E-5 - 1.5206676508978484E-4	th(alpha)=17.251077 > -82.539847;th'(alpha)=-179.007852 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 3.801669127244621E-5 - 7.603338254489242E-5	th(alpha)=16.764958 > -40.046710;th'(alpha)=-8.030894 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 1.9008345636223105E-5 - 3.801669127244621E-5	th(alpha)=16.421346 > -18.800141;th'(alpha)=628.699032 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 9.504172818111552E-6 - 1.9008345636223105E-5	th(alpha)=16.035717 > -8.176857;th'(alpha)=579.762074 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 4.752086409055776E-6 - 9.504172818111552E-6	th(alpha)=15.520523 > -2.865215;th'(alpha)=1105.419385 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 2.376043204527888E-6 - 4.752086409055776E-6	th(alpha)=14.663695 > -0.209394;th'(alpha)=1983.892916 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 1.188021602263944E-6 - 2.376043204527888E-6	th(alpha)=13.272854 > 1.118517;th'(alpha)=3641.844368 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 5.94010801131972E-7 - 1.188021602263944E-6	th(alpha)=11.031601 > 1.782472;th'(alpha)=5682.911117 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 2.97005400565986E-7 - 5.94010801131972E-7	th(alpha)=7.813998 > 2.114450;th'(alpha)=8029.343592 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 1.48502700282993E-7 - 2.97005400565986E-7	th(alpha)=4.471362 > 2.280439;th'(alpha)=8425.231119 >= -89419958956.748580ARMIJO: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 7.42513501414965E-8 - 1.48502700282993E-7	th(alpha)=2.537923 > 2.363433;th'(alpha)=5492.063823 >= -89419958956.748580END: th(0)=2.446427;th'(0)=-111774948695.935710;	0.0 - 3.712567507074825E-8 - 7.42513501414965E-8	th(alpha)=1.826965;th'(alpha)=3223.910697Iteration 1 complete. Error: 1.8269650832611566ARMIJO: th(0)=1.826226;th'(0)=-208727390686.197480;	0.0 - 4.624860208974361E-8 - Infinity	th(alpha)=4.055818 > 1.729692;th'(alpha)=14252.660262 >= -166981912548.958000ARMIJO: th(0)=1.826226;th'(0)=-208727390686.197480;	0.0 - 2.3124301044871804E-8 - 4.624860208974361E-8	th(alpha)=2.410016 > 1.777959;th'(alpha)=13271.912556 >= -166981912548.958000END: th(0)=1.826226;th'(0)=-208727390686.197480;	0.0 - 1.1562150522435902E-8 - 2.3124301044871804E-8	th(alpha)=1.496206;th'(alpha)=10541.654485Iteration 2 complete. Error: 1.4962059003241264END: th(0)=1.498305;th'(0)=-89005097875.392550;	0.0 - 1.4403328634290123E-8 - Infinity	th(alpha)=1.225824;th'(alpha)=5148.963264Iteration 3 complete. Error: 1.2258242717899441END: th(0)=1.231176;th'(0)=-41349380794.516594;	0.0 - 1.7942672113185322E-8 - Infinity	th(alpha)=1.197980;th'(alpha)=5327.152534Iteration 4 complete. Error: 1.1979798725716564ARMIJO: th(0)=1.198620;th'(0)=-79259433336.708010;	0.0 - 2.235174179077149E-8 - Infinity	th(alpha)=1.669256 > 1.180904;th'(alpha)=8410.354052 >= -63407546669.366410END: th(0)=1.198620;th'(0)=-79259433336.708010;	0.0 - 1.1175870895385745E-8 - 2.235174179077149E-8	th(alpha)=1.097879;th'(alpha)=4942.595661Iteration 5 complete. Error: 1.0978793605865733WOLFE: th(0)=1.097500;th'(0)=-214.721735;	0.0 - 1.3922128151530598E-8 - Infinity	th(alpha)=1.097500 <= 1.097500;th'(alpha)=-219.840493 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	1.3922128151530598E-8 - 2.7844256303061195E-8 - Infinity	th(alpha)=1.097500 <= 1.097500;th'(alpha)=-223.748254 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	2.7844256303061195E-8 - 5.568851260612239E-8 - Infinity	th(alpha)=1.097500 <= 1.097500;th'(alpha)=-219.404082 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	5.568851260612239E-8 - 1.1137702521224478E-7 - Infinity	th(alpha)=1.097500 <= 1.097500;th'(alpha)=-217.087468 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	1.1137702521224478E-7 - 2.2275405042448956E-7 - Infinity	th(alpha)=1.097499 <= 1.097500;th'(alpha)=-213.210585 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	2.2275405042448956E-7 - 4.455081008489791E-7 - Infinity	th(alpha)=1.097499 <= 1.097500;th'(alpha)=-219.207529 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	4.455081008489791E-7 - 8.910162016979582E-7 - Infinity	th(alpha)=1.097499 <= 1.097500;th'(alpha)=-215.708589 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	8.910162016979582E-7 - 1.7820324033959165E-6 - Infinity	th(alpha)=1.097499 <= 1.097500;th'(alpha)=-213.660960 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	1.7820324033959165E-6 - 3.564064806791833E-6 - Infinity	th(alpha)=1.097499 <= 1.097500;th'(alpha)=-207.669943 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	3.564064806791833E-6 - 7.128129613583666E-6 - Infinity	th(alpha)=1.097498 <= 1.097500;th'(alpha)=-214.559648 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	7.128129613583666E-6 - 1.4256259227167332E-5 - Infinity	th(alpha)=1.097496 <= 1.097500;th'(alpha)=-218.300996 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	1.4256259227167332E-5 - 2.8512518454334664E-5 - Infinity	th(alpha)=1.097493 <= 1.097499;th'(alpha)=-201.206281 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	2.8512518454334664E-5 - 5.702503690866933E-5 - Infinity	th(alpha)=1.097487 <= 1.097499;th'(alpha)=-219.426308 < -171.777388WOLFE: th(0)=1.097500;th'(0)=-214.721735;	5.702503690866933E-5 - 1.1405007381733866E-4 - Infinity	th(alpha)=1.097474 <= 1.097499;th'(alpha)=-201.140160 < -171.777388WOLFE: th(0)=
    ...skipping 234065 bytes...
    omplete. Error: 0.35096017628609344WOLFE: th(0)=0.350728;th'(0)=-9.771692;	0.0 - 0.029425497712598128 - Infinity	th(alpha)=0.350516 <= 0.350728;th'(alpha)=-10.035792 < -9.577405END: th(0)=0.350728;th'(0)=-9.771692;	0.029425497712598128 - 0.058850995425196256 - Infinity	th(alpha)=0.350318;th'(alpha)=-9.135730Iteration 428 complete. Error: 0.3503184786889512END: th(0)=0.350087;th'(0)=-9.452659;	0.0 - 0.07331250582833825 - Infinity	th(alpha)=0.349648;th'(alpha)=-0.570602Iteration 429 complete. Error: 0.3496484023848661END: th(0)=0.349417;th'(0)=-3.987852;	0.0 - 0.0913276567711039 - Infinity	th(alpha)=0.349152;th'(alpha)=-0.355484Iteration 430 complete. Error: 0.34915199456521084END: th(0)=0.348921;th'(0)=-12.455961;	0.0 - 0.11376968768235073 - Infinity	th(alpha)=0.348034;th'(alpha)=-1.123055Iteration 431 complete. Error: 0.34803431264493845END: th(0)=0.354868;th'(0)=-5.161920;	0.0 - 0.14172641993629873 - Infinity	th(alpha)=0.354386;th'(alpha)=-0.564453Iteration 432 complete. Error: 0.3543861428983203END: th(0)=0.355599;th'(0)=-4.827057;	0.0 - 0.1765529862755888 - Infinity	th(alpha)=0.355051;th'(alpha)=-0.758807Iteration 433 complete. Error: 0.35505078176449356END: th(0)=0.354816;th'(0)=-8.547381;	0.0 - 0.21993751748501478 - Infinity	th(alpha)=0.353630;th'(alpha)=-1.271605Iteration 434 complete. Error: 0.35363019314543004END: th(0)=0.353397;th'(0)=-4.295429;	0.0 - 0.27398297031331176 - Infinity	th(alpha)=0.352706;th'(alpha)=-0.785749Iteration 435 complete. Error: 0.35270613834181125END: th(0)=0.352474;th'(0)=-5.453272;	0.0 - 0.3413090630470523 - Infinity	th(alpha)=0.351334;th'(alpha)=-1.434760Iteration 436 complete. Error: 0.3513343072464027END: th(0)=0.351103;th'(0)=-6.559203;	0.0 - 0.4251792598088963 - Infinity	th(alpha)=0.349583;th'(alpha)=-1.574127Iteration 437 complete. Error: 0.3495830455618308END: th(0)=0.349353;th'(0)=-5.513931;	0.0 - 0.5296589588267666 - Infinity	th(alpha)=0.347774;th'(alpha)=-1.713998Iteration 438 complete. Error: 0.34777427069626926END: th(0)=0.347546;th'(0)=-5.626191;	0.0 - 0.6598125524550447 - Infinity	th(alpha)=0.345424;th'(alpha)=-2.430890Iteration 439 complete. Error: 0.3454241268027054ARMIJO: th(0)=0.346794;th'(0)=-13.991893;	0.0 - 0.8219489109399356 - Infinity	th(alpha)=0.356651 > 0.346790;th'(alpha)=47.631330 >= -13.713698END: th(0)=0.346794;th'(0)=-13.991893;	0.0 - 0.4109744554699678 - 0.8219489109399356	th(alpha)=0.346142;th'(alpha)=6.232288Iteration 440 complete. Error: 0.3461419783169912END: th(0)=0.345915;th'(0)=-8.296553;	0.0 - 0.5119635945705786 - Infinity	th(alpha)=0.343847;th'(alpha)=-1.742810Iteration 441 complete. Error: 0.34384727659254116END: th(0)=0.343622;th'(0)=-2.633852;	0.0 - 0.6377688897133447 - Infinity	th(alpha)=0.342713;th'(alpha)=-0.839359Iteration 442 complete. Error: 0.3427133963050572END: th(0)=0.342489;th'(0)=-1.572762;	0.0 - 0.7944884382401501 - Infinity	th(alpha)=0.341857;th'(alpha)=-0.583998Iteration 443 complete. Error: 0.3418568106928244END: th(0)=0.341633;th'(0)=-1.162155;	0.0 - 0.9897188286825672 - Infinity	th(alpha)=0.341099;th'(alpha)=-0.260137Iteration 444 complete. Error: 0.3410994705873528END: th(0)=0.342468;th'(0)=-0.552133;	0.0 - 1.2329233664099037 - Infinity	th(alpha)=0.342265;th'(alpha)=-0.027357Iteration 445 complete. Error: 0.34226451415167697END: th(0)=0.342041;th'(0)=-0.176323;	0.0 - 1.5358907837117362 - Infinity	th(alpha)=0.341964;th'(alpha)=-0.043802Iteration 446 complete. Error: 0.34196403394753END: th(0)=0.342724;th'(0)=-0.372011;	0.0 - 1.9133066691400344 - Infinity	th(alpha)=0.342358;th'(alpha)=-0.320853Iteration 447 complete. Error: 0.3423584725042937ARMIJO: th(0)=0.342135;th'(0)=-0.553323;	0.0 - 2.3834653147204508 - Infinity	th(alpha)=0.342236 > 0.342135;th'(alpha)=2.002797 >= -0.542322END: th(0)=0.342135;th'(0)=-0.553323;	0.0 - 1.1917326573602254 - 2.3834653147204508	th(alpha)=0.341899;th'(alpha)=0.030464Iteration 448 complete. Error: 0.34189912857630794END: th(0)=0.341676;th'(0)=-0.361235;	0.0 - 1.484578243023851 - Infinity	th(alpha)=0.341528;th'(alpha)=-0.000448Iteration 449 complete. Error: 0.34152838871790325END: th(0)=0.341306;th'(0)=-0.416344;	0.0 - 1.8493850496148558 - Infinity	th(alpha)=0.341228;th'(alpha)=0.280958Iteration 450 complete. Error: 0.34122815528994804END: th(0)=0.341006;th'(0)=-0.379896;	0.0 - 2.3038361755676044 - Infinity	th(alpha)=0.340789;th'(alpha)=0.032853Iteration 451 complete. Error: 0.34078921401107615END: th(0)=0.348395;th'(0)=-1.305157;	0.0 - 2.869960003710052 - Infinity	th(alpha)=0.347431;th'(alpha)=2.278029Iteration 452 complete. Error: 0.34743073549873604ARMIJO: th(0)=0.354037;th'(0)=-6.132625;	0.0 - 3.575197972080677 - Infinity	th(alpha)=0.354861 > 0.354028;th'(alpha)=47.055115 >= -6.010693END: th(0)=0.354037;th'(0)=-6.132625;	0.0 - 1.7875989860403385 - 3.575197972080677	th(alpha)=0.349558;th'(alpha)=-2.265298Iteration 453 complete. Error: 0.3495582972003739ARMIJO: th(0)=0.351004;th'(0)=-17.636351;	0.0 - 2.226867364535777 - Infinity	th(alpha)=0.355860 > 0.350989;th'(alpha)=38.220584 >= -17.285694END: th(0)=0.351004;th'(0)=-17.636351;	0.0 - 1.1134336822678885 - 2.226867364535777	th(alpha)=0.346861;th'(alpha)=7.168726Iteration 454 complete. Error: 0.34686121952871674END: th(0)=0.346636;th'(0)=-16.981282;	0.0 - 1.3870387872111423 - Infinity	th(alpha)=0.343772;th'(alpha)=4.226064Iteration 455 complete. Error: 0.34377156437553297ARMIJO: th(0)=0.343549;th'(0)=-8.510091;	0.0 - 1.727877131675704 - Infinity	th(alpha)=0.343896 > 0.343543;th'(alpha)=9.213003 >= -8.340888END: th(0)=0.343549;th'(0)=-8.510091;	0.0 - 0.863938565837852 - 1.727877131675704	th(alpha)=0.341979;th'(alpha)=1.047587Iteration 456 complete. Error: 0.3419788380952295END: th(0)=0.341757;th'(0)=-0.977992;	0.0 - 1.0762350013912698 - Infinity	th(alpha)=0.341359;th'(alpha)=-0.283455Iteration 457 complete. Error: 0.3413585817907834END: th(0)=0.342715;th'(0)=-0.272845;	0.0 - 1.3406992395302542 - Infinity	th(alpha)=0.342544;th'(alpha)=-0.124754Iteration 458 complete. Error: 0.34254356608012626ARMIJO: th(0)=0.342322;th'(0)=-0.365737;	0.0 - 1.6701505234018332 - Infinity	th(alpha)=0.342386 > 0.342322;th'(alpha)=0.621485 >= -0.358465END: th(0)=0.342322;th'(0)=-0.365737;	0.0 - 0.8350752617009166 - 1.6701505234018332	th(alpha)=0.342258;th'(alpha)=0.014239Iteration 459 complete. Error: 0.34225846112885977ARMIJO: th(0)=0.352734;th'(0)=-9.056091;	0.0 - 1.040279090408357 - Infinity	th(alpha)=1.028973 > 0.352730;th'(alpha)=2453.948702 >= -8.876032ARMIJO: th(0)=0.352734;th'(0)=-9.056091;	0.0 - 0.5201395452041785 - 1.040279090408357	th(alpha)=0.459427 > 0.352732;th'(alpha)=403.490499 >= -8.876032ARMIJO: th(0)=0.352734;th'(0)=-9.056091;	0.0 - 0.26006977260208924 - 0.5201395452041785	th(alpha)=0.367450 > 0.352733;th'(alpha)=63.194471 >= -8.876032ARMIJO: th(0)=0.352734;th'(0)=-9.056091;	0.0 - 0.13003488630104462 - 0.26006977260208924	th(alpha)=0.354022 > 0.352734;th'(alpha)=7.128106 >= -8.876032END: th(0)=0.352734;th'(0)=-9.056091;	0.0 - 0.06501744315052231 - 0.13003488630104462	th(alpha)=0.352651;th'(alpha)=0.344421Iteration 460 complete. Error: 0.35265134712656976END: th(0)=0.353997;th'(0)=-2.165224;	0.0 - 0.08099424054729865 - Infinity	th(alpha)=0.353959;th'(alpha)=0.080408Iteration 461 complete. Error: 0.353958913283705END: th(0)=0.353730;th'(0)=-0.923910;	0.0 - 0.10089703138043159 - Infinity	th(alpha)=0.353670;th'(alpha)=-0.798698Iteration 462 complete. Error: 0.3536702346665331END: th(0)=0.353442;th'(0)=-1.270750;	0.0 - 0.12569055370596138 - Infinity	th(alpha)=0.353340;th'(alpha)=-0.128365Iteration 463 complete. Error: 0.35333961559852245END: th(0)=0.354682;th'(0)=-0.814746;	0.0 - 0.15657661156892191 - Infinity	th(alpha)=0.354599;th'(alpha)=-0.115011Iteration 464 complete. Error: 0.3545988848663245END: th(0)=0.354370;th'(0)=-4.868864;	0.0 - 0.19505232945156697 - Infinity	th(alpha)=0.353917;th'(alpha)=-0.425768Iteration 465 complete. Error: 0.3539168215986883END: th(0)=0.353688;th'(0)=-0.823074;	0.0 - 0.242982721641896 - Infinity	th(alpha)=0.353564;th'(alpha)=-0.172727Iteration 466 complete. Error: 0.35356372711040773END: th(0)=0.353336;th'(0)=-1.559875;	0.0 - 0.3026910941412948 - Infinity	th(alpha)=0.353093;th'(alpha)=-0.303145Iteration 467 complete. Error: 0.3530926610935297
```

Code from [MNistDemo.java:60](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L60) executed in 0.53 seconds: 
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
    76.49000000000001
```



Code from [MNistDemo.java:73](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L73) executed in 0.73 seconds: 
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
![[9]](etc/bellsAndWhistles.1.png)   | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[5]](etc/bellsAndWhistles.2.png)   | 2 (58.8%), 4 (41.2%), 6 (0.1%) 
![[9]](etc/bellsAndWhistles.3.png)   | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[6]](etc/bellsAndWhistles.4.png)   | 2 (99.8%), 6 (0.2%), 0 (0.0%)  
![[9]](etc/bellsAndWhistles.5.png)   | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[5]](etc/bellsAndWhistles.6.png)   | 3 (99.9%), 5 (0.1%), 4 (0.0%)  
![[9]](etc/bellsAndWhistles.7.png)   | 7 (42.2%), 0 (6.4%), 1 (6.4%)  
![[9]](etc/bellsAndWhistles.8.png)   | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[3]](etc/bellsAndWhistles.9.png)   | 5 (84.8%), 3 (14.3%), 6 (0.2%) 
![[6]](etc/bellsAndWhistles.10.png)  | 2 (100.0%), 6 (0.0%), 0 (0.0%) 
![[9]](etc/bellsAndWhistles.11.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[8]](etc/bellsAndWhistles.12.png)  | 2 (99.6%), 5 (0.4%), 0 (0.0%)  
![[9]](etc/bellsAndWhistles.13.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[3]](etc/bellsAndWhistles.14.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[7]](etc/bellsAndWhistles.15.png)  | 2 (90.3%), 7 (9.7%), 3 (0.0%)  
![[6]](etc/bellsAndWhistles.16.png)  | 2 (100.0%), 1 (0.0%), 0 (0.0%) 
![[9]](etc/bellsAndWhistles.17.png)  | 7 (99.8%), 0 (0.0%), 1 (0.0%)  
![[9]](etc/bellsAndWhistles.18.png)  | 8 (11.7%), 0 (9.8%), 1 (9.8%)  
![[7]](etc/bellsAndWhistles.19.png)  | 5 (29.5%), 0 (7.8%), 1 (7.8%)  
![[9]](etc/bellsAndWhistles.20.png)  | 8 (17.3%), 0 (9.2%), 1 (9.2%)  
![[9]](etc/bellsAndWhistles.21.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.22.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.23.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.24.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.25.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.26.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[2]](etc/bellsAndWhistles.27.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[7]](etc/bellsAndWhistles.28.png)  | 4 (100.0%), 8 (0.0%), 0 (0.0%) 
![[9]](etc/bellsAndWhistles.29.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[2]](etc/bellsAndWhistles.30.png)  | 8 (99.9%), 0 (0.0%), 1 (0.0%)  
![[9]](etc/bellsAndWhistles.31.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.32.png)  | 7 (99.7%), 2 (0.0%), 0 (0.0%)  
![[4]](etc/bellsAndWhistles.33.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[5]](etc/bellsAndWhistles.34.png)  | 3 (71.1%), 6 (7.0%), 0 (2.7%)  
![[2]](etc/bellsAndWhistles.35.png)  | 3 (100.0%), 2 (0.0%), 8 (0.0%) 
![[9]](etc/bellsAndWhistles.36.png)  | 4 (69.1%), 0 (3.4%), 1 (3.4%)  
![[1]](etc/bellsAndWhistles.37.png)  | 5 (48.9%), 1 (47.9%), 6 (1.0%) 
![[9]](etc/bellsAndWhistles.38.png)  | 3 (96.5%), 4 (2.9%), 0 (0.1%)  
![[3]](etc/bellsAndWhistles.39.png)  | 8 (87.1%), 2 (3.2%), 0 (1.2%)  
![[9]](etc/bellsAndWhistles.40.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.41.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[5]](etc/bellsAndWhistles.42.png)  | 7 (99.7%), 5 (0.1%), 0 (0.0%)  
![[9]](etc/bellsAndWhistles.43.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.44.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[6]](etc/bellsAndWhistles.45.png)  | 5 (100.0%), 8 (0.0%), 0 (0.0%) 
![[3]](etc/bellsAndWhistles.46.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[8]](etc/bellsAndWhistles.47.png)  | 7 (81.7%), 0 (2.0%), 1 (2.0%)  
![[9]](etc/bellsAndWhistles.48.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[5]](etc/bellsAndWhistles.49.png)  | 8 (65.6%), 5 (34.4%), 3 (0.0%) 
![[9]](etc/bellsAndWhistles.50.png)  | 5 (44.7%), 0 (6.1%), 1 (6.1%)  
![[7]](etc/bellsAndWhistles.51.png)  | 4 (92.3%), 7 (5.5%), 3 (2.2%)  
![[3]](etc/bellsAndWhistles.52.png)  | 5 (100.0%), 0 (0.0%), 1 (0.0%) 
![[4]](etc/bellsAndWhistles.53.png)  | 6 (99.9%), 0 (0.0%), 1 (0.0%)  
![[4]](etc/bellsAndWhistles.54.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.55.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[8]](etc/bellsAndWhistles.56.png)  | 1 (100.0%), 2 (0.0%), 8 (0.0%) 
![[9]](etc/bellsAndWhistles.57.png)  | 3 (80.8%), 0 (2.1%), 1 (2.1%)  
![[8]](etc/bellsAndWhistles.58.png)  | 4 (100.0%), 5 (0.0%), 8 (0.0%) 
![[9]](etc/bellsAndWhistles.59.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.60.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[8]](etc/bellsAndWhistles.61.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.62.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.63.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[5]](etc/bellsAndWhistles.64.png)  | 6 (78.9%), 4 (20.0%), 5 (1.1%) 
![[8]](etc/bellsAndWhistles.65.png)  | 4 (40.8%), 0 (6.6%), 1 (6.6%)  
![[9]](etc/bellsAndWhistles.66.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[4]](etc/bellsAndWhistles.67.png)  | 6 (100.0%), 0 (0.0%), 1 (0.0%) 
![[7]](etc/bellsAndWhistles.68.png)  | 2 (99.0%), 7 (0.4%), 0 (0.1%)  
![[4]](etc/bellsAndWhistles.69.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[7]](etc/bellsAndWhistles.70.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[3]](etc/bellsAndWhistles.71.png)  | 5 (99.9%), 4 (0.1%), 0 (0.0%)  
![[9]](etc/bellsAndWhistles.72.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[5]](etc/bellsAndWhistles.73.png)  | 1 (93.7%), 5 (6.0%), 8 (0.3%)  
![[2]](etc/bellsAndWhistles.74.png)  | 3 (99.8%), 2 (0.2%), 0 (0.0%)  
![[5]](etc/bellsAndWhistles.75.png)  | 4 (100.0%), 5 (0.0%), 6 (0.0%) 
![[9]](etc/bellsAndWhistles.76.png)  | 7 (100.0%), 8 (0.0%), 1 (0.0%) 
![[2]](etc/bellsAndWhistles.77.png)  | 7 (100.0%), 8 (0.0%), 0 (0.0%) 
![[9]](etc/bellsAndWhistles.78.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.79.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[5]](etc/bellsAndWhistles.80.png)  | 1 (97.9%), 0 (0.2%), 2 (0.2%)  
![[6]](etc/bellsAndWhistles.81.png)  | 2 (69.7%), 4 (28.3%), 6 (0.3%) 
![[8]](etc/bellsAndWhistles.82.png)  | 5 (99.7%), 8 (0.2%), 6 (0.0%)  
![[3]](etc/bellsAndWhistles.83.png)  | 7 (100.0%), 3 (0.0%), 0 (0.0%) 
![[5]](etc/bellsAndWhistles.84.png)  | 3 (99.8%), 0 (0.2%), 5 (0.0%)  
![[7]](etc/bellsAndWhistles.85.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.86.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[2]](etc/bellsAndWhistles.87.png)  | 7 (100.0%), 3 (0.0%), 0 (0.0%) 
![[5]](etc/bellsAndWhistles.88.png)  | 8 (100.0%), 5 (0.0%), 3 (0.0%) 
![[4]](etc/bellsAndWhistles.89.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[6]](etc/bellsAndWhistles.90.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.91.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.92.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[5]](etc/bellsAndWhistles.93.png)  | 8 (93.9%), 5 (6.1%), 3 (0.0%)  
![[8]](etc/bellsAndWhistles.94.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[4]](etc/bellsAndWhistles.95.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.96.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.97.png)  | 7 (18.4%), 0 (9.1%), 1 (9.1%)  
![[9]](etc/bellsAndWhistles.98.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[9]](etc/bellsAndWhistles.99.png)  | 0 (10.0%), 1 (10.0%), 2 (10.0%)
![[8]](etc/bellsAndWhistles.100.png) | 4 (100.0%), 8 (0.0%), 0 (0.0%) 




