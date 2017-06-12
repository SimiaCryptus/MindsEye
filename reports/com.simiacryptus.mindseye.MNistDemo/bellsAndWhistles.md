First, define a model:

This is a very simple model that performs basic logistic regression. It is expected to be trainable to about 91% accuracy on MNIST.

Code from [MNistDemo.java:137](../.././src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L137) executed in 0.01 seconds: 
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
    PipelineNetwork/fcd30a0f-ce26-4a55-aa93-165a00000007
```



We use the standard MNIST dataset, made available by a helper function. In order to use data, we convert it into data tensors; helper functions are defined to work with images.

Code from [MNistDemo.java:120](../.././src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L120) executed in 0.88 seconds: 
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
    [[Lcom.simiacryptus.util.ml.Tensor;@6535117e
```



Code from [MNistDemo.java:169](../.././src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L169) executed in 300.68 seconds: 
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
    trainer.setLineSearchFactory(()->new ArmijoWolfeConditions().setC1(1e-4).setC2(0.9));
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
    Constructing line search parameters: GD+Trust
    ARMIJO: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 2.154434690031884 - Infinity	th(alpha)=28130.943396 > -36152997.431146;th'(alpha)=0.000000 >= -151026624904.919200ARMIJO: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 1.077217345015942 - 2.154434690031884	th(alpha)=14074.700854 > -18076497.441503;th'(alpha)=0.000000 >= -151026624904.919200ARMIJO: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 0.3590724483386473 - 1.077217345015942	th(alpha)=4703.872494 > -6025497.448407;th'(alpha)=0.000000 >= -151026624904.919200ARMIJO: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 0.08976811208466183 - 0.3590724483386473	th(alpha)=1189.811858 > -1506372.450996;th'(alpha)=0.000000 >= -151026624904.919200ARMIJO: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 0.017953622416932366 - 0.08976811208466183	th(alpha)=252.729022 > -301272.451686;th'(alpha)=0.000000 >= -151026624904.919200ARMIJO: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 0.002992270402822061 - 0.017953622416932366	th(alpha)=57.503431 > -50209.951830;th'(alpha)=0.000000 >= -151026624904.919200ARMIJO: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 4.2746720040315154E-4 - 0.002992270402822061	th(alpha)=24.036188 > -7170.666140;th'(alpha)=3.601790 >= -151026624904.919200ARMIJO: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 5.343340005039394E-5 - 4.2746720040315154E-4	th(alpha)=19.132312 > -894.103644;th'(alpha)=5.794354 >= -151026624904.919200ARMIJO: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 5.9370444500437714E-6 - 5.343340005039394E-5	th(alpha)=18.306178 > -97.079835;th'(alpha)=139.960135 >= -151026624904.919200ARMIJO: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 5.937044450043771E-7 - 5.9370444500437714E-6	th(alpha)=15.912144 > -7.414656;th'(alpha)=2742.157741 >= -151026624904.919200ARMIJO: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 5.397313136403428E-8 - 5.937044450043771E-7	th(alpha)=4.280517 > 1.642433;th'(alpha)=3616.897241 >= -151026624904.919200END: th(0)=2.548141;th'(0)=-167807361005.465760;	0.0 - 4.4977609470028565E-9 - 5.397313136403428E-8	th(alpha)=2.094634;th'(alpha)=-191.270830Iteration 1 complete. Error: 2.0946343221108705END: th(0)=2.094892;th'(0)=-61781786268.377920;	0.0 - 9.690132211693611E-9 - Infinity	th(alpha)=1.602651;th'(alpha)=-362.734759Iteration 2 complete. Error: 1.6026509831392544END: th(0)=1.603672;th'(0)=-37400405358.896210;	0.0 - 2.0876756987868097E-8 - Infinity	th(alpha)=1.136853;th'(alpha)=-205.163752Iteration 3 complete. Error: 1.136853495652726ARMIJO: th(0)=1.136524;th'(0)=-37427820771.785164;	0.0 - 4.497760947002857E-8 - Infinity	th(alpha)=3.049645 > 0.968182;th'(alpha)=3679.764976 >= -33685038694.606647ARMIJO: th(0)=1.136524;th'(0)=-37427820771.785164;	0.0 - 2.2488804735014283E-8 - 4.497760947002857E-8	th(alpha)=1.396614 > 1.052353;th'(alpha)=1154.858965 >= -33685038694.606647END: th(0)=1.136524;th'(0)=-37427820771.785164;	0.0 - 7.49626824500476E-9 - 2.2488804735014283E-8	th(alpha)=0.985728;th'(alpha)=-24.652476Iteration 4 complete. Error: 0.9857283915639307END: th(0)=0.986644;th'(0)=-11830865626.335201;	0.0 - 1.6150220352822687E-8 - Infinity	th(alpha)=0.877605;th'(alpha)=-47.756198Iteration 5 complete. Error: 0.8776052066132306Constructing line search parameters: LBFGS+Trust
    END: th(0)=0.878320;th'(0)=-331.839146;	0.0 - 2.154434690031884 - Infinity	th(alpha)=0.545302;th'(alpha)=-103.863151Iteration 6 complete. Error: 0.5453019530863209ARMIJO: th(0)=0.544823;th'(0)=-177.483856;	0.0 - 4.641588833612779 - Infinity	th(alpha)=1.462636 > 0.462443;th'(alpha)=1792.611615 >= -159.735470ARMIJO: th(0)=0.544823;th'(0)=-177.483856;	0.0 - 2.3207944168063896 - 4.641588833612779	th(alpha)=0.713217 > 0.503633;th'(alpha)=522.091418 >= -159.735470END: th(0)=0.544823;th'(0)=-177.483856;	0.0 - 0.7735981389354633 - 2.3207944168063896	th(alpha)=0.496628;th'(alpha)=16.941826Iteration 7 complete. Error: 0.4966283579982872END: th(0)=0.496744;th'(0)=-69.231349;	0.0 - 1.666666666666667 - Infinity	th(alpha)=0.426182;th'(alpha)=-37.334006Iteration 8 complete. Error: 0.426182197919002END: th(0)=0.425766;th'(0)=-62.937572;	0.0 - 3.5907244833864738 - Infinity	th(alpha)=0.360211;th'(alpha)=30.343807Iteration 9 complete. Error: 0.3602114511625476ARMIJO: th(0)=0.359915;th'(0)=-67.145251;	0.0 - 7.735981389354634 - Infinity	th(alpha)=0.770420 > 0.307971;th'(alpha)=864.224182 >= -60.430726ARMIJO: th(0)=0.359915;th'(0)=-67.145251;	0.0 - 3.867990694677317 - 7.735981389354634	th(alpha)=0.414989 > 0.333943;th'(alpha)=239.755922 >= -60.430726END: th(0)=0.359915;th'(0)=-67.145251;	0.0 - 1.2893302315591055 - 3.867990694677317	th(alpha)=0.324093;th'(alpha)=1.039856Iteration 10 complete. Error: 0.3240929239596477WOLFE: th(0)=0.323777;th'(0)=-13.716004;	0.0 - 2.7777777777777786 - Infinity	th(alpha)=0.297160 <= 0.319967;th'(alpha)=-19.858857 < -12.344404WOLFE: th(0)=0.323777;th'(0)=-13.716004;	2.7777777777777786 - 5.555555555555557 - Infinity	th(alpha)=0.283025 <= 0.316157;th'(alpha)=-16.614883 < -12.344404ARMIJO: th(0)=0.323777;th'(0)=-13.716004;	5.555555555555557 - 16.66666666666667 - Infinity	th(alpha)=0.336287 > 0.300917;th'(alpha)=189.138418 >= -12.344404END: th(0)=0.323777;th'(0)=-13.716004;	5.555555555555557 - 11.111111111111114 - 16.66666666666667	th(alpha)=0.289214;th'(alpha)=56.166366Iteration 11 complete. Error: 0.2892137193108815ARMIJO: th(0)=0.288948;th'(0)=-67.859941;	0.0 - 23.938163222576495 - Infinity	th(alpha)=3.253005 > 0.126504;th'(alpha)=3863.171024 >= -61.073947ARMIJO: th(0)=0.288948;th'(0)=-67.859941;	0.0 - 11.969081611288248 - 23.938163222576495	th(alpha)=1.375368 > 0.207726;th'(alpha)=1782.251565 >= -61.073947ARMIJO: th(0)=0.288948;th'(0)=-67.859941;	0.0 - 3.989693870429416 - 11.969081611288248	th(alpha)=0.344757 > 0.261874;th'(alpha)=307.657733 >= -61.073947END: th(0)=0.288948;th'(0)=-67.859941;	0.0 - 0.997423467607354 - 3.989693870429416	th(alpha)=0.251943;th'(alpha)=-15.297586Iteration 12 complete. Error: 0.2519430542460411END: th(0)=0.252672;th'(0)=-22.833204;	0.0 - 2.1488837192651764 - Infinity	th(alpha)=0.226581;th'(alpha)=-3.398828Iteration 13 complete. Error: 0.2265812923097165END: th(0)=0.226517;th'(0)=-25.179199;	0.0 - 4.6296296296296315 - Infinity	th(alpha)=0.192632;th'(alpha)=49.609219Iteration 14 complete. Error: 0.1926320964286099ARMIJO: th(0)=0.192912;th'(0)=-57.023395;	0.0 - 9.97423467607354 - Infinity	th(alpha)=0.689945 > 0.136036;th'(alpha)=1163.825824 >= -51.321055ARMIJO: th(0)=0.192912;th'(0)=-57.023395;	0.0 - 4.98711733803677 - 9.97423467607354	th(alpha)=0.254802 > 0.164474;th'(alpha)=260.343473 >= -51.321055END: th(0)=0.192912;th'(0)=-57.023395;	0.0 - 1.6623724460122566 - 4.98711733803677	th(alpha)=0.156911;th'(alpha)=1.828598Iteration 15 complete. Error: 0.15691093654816257END: th(0)=0.156773;th'(0)=-10.645661;	0.0 - 3.5814728654419605 - Infinity	th(alpha)=0.136272;th'(alpha)=-5.678105Iteration 16 complete. Error: 0.13627223630015844END: th(0)=0.136147;th'(0)=-30.949019;	0.0 - 7.716049382716053 - Infinity	th(alpha)=0.110799;th'(alpha)=79.392027Iteration 17 complete. Error: 0.11079857469568549ARMIJO: th(0)=0.114769;th'(0)=-153.420286;	0.0 - 16.623724460122567 - Infinity	th(alpha)=2.623089 > -0.140272;th'(alpha)=2286.134769 >= -138.078258ARMIJO: th(0)=0.114769;th'(0)=-153.420286;	0.0 - 8.311862230061283 - 16.623724460122567	th(alpha)=1.166529 > -0.012751;th'(alpha)=1720.198093 >= -138.078258ARMIJO: th(0)=0.114769;th'(0)=-153.420286;	0.0 - 2.7706207433537613 - 8.311862230061283	th(alpha)=0.188085 > 0.072263;th'(alpha)=286.215114 >= -138.078258END: th(0)=0.114769;th'(0)=-153.420286;	0.0 - 0.6926551858384403 - 2.7706207433537613	th(alpha)=0.075188;th'(alpha)=-8.306247Iteration 18 complete. Error: 0.07518836722701232END: th(0)=0.075123;th'(0)=-30.678809;	0.0 - 1.492280360600817 - Infinity	th(alpha)=0.051488;th'(alpha)=-12.644908Iteration 19 complete. Error: 0.051487822447910385END: th(0)=0.051445;th'(0)=-26.587309;	0.0 - 3.215020576131689 - Infinity	th(alpha)=0.042329;th'(alpha)=34.868577Iteration 20 complete. Error: 0.04232865052918268ARMIJO: th(0)=0.042296;th'(0)=-26.581142;	0.0 - 6.926551858384404 - Infinity	th(alpha)=0.153040 > 0.023884;th'(alpha)=397.202965 >= -23
    ...skipping 283451 bytes...
    (0)=-0.036966;	0.0 - 9.40225237976971E-9 - 1.034247761774668E-7	th(alpha)=0.065527 > 0.065527;th'(alpha)=-0.036329 >= -0.036965ARMIJO: th(0)=0.065527;th'(0)=-0.036966;	0.0 - 7.835210316474758E-10 - 9.40225237976971E-9	th(alpha)=0.065527 > 0.065527;th'(alpha)=-0.035585 >= -0.036965ARMIJO: th(0)=0.065527;th'(0)=-0.036966;	0.0 - 6.027084858826737E-11 - 7.835210316474758E-10	th(alpha)=0.065527 > 0.065527;th'(alpha)=-0.038043 >= -0.036965ARMIJO: th(0)=0.065527;th'(0)=-0.036966;	0.0 - 4.305060613447669E-12 - 6.027084858826737E-11	th(alpha)=0.065527 > 0.065527;th'(alpha)=-0.033794 >= -0.036965END: th(0)=0.065527;th'(0)=-0.036966;	0.0 - 2.870040408965113E-13 - 4.305060613447669E-12	th(alpha)=0.065527;th'(alpha)=-0.036782Iteration 769 complete. Error: 0.065527412177737ARMIJO: th(0)=0.067587;th'(0)=-0.226217;	0.0 - 6.183314618867734E-13 - Infinity	th(alpha)=0.067587 > 0.067587;th'(alpha)=-0.225194 >= -0.226212ARMIJO: th(0)=0.067587;th'(0)=-0.226217;	0.0 - 3.091657309433867E-13 - 6.183314618867734E-13	th(alpha)=0.067587 > 0.067587;th'(alpha)=-0.230087 >= -0.226212END: th(0)=0.067587;th'(0)=-0.226217;	0.0 - 1.0305524364779557E-13 - 3.091657309433867E-13	th(alpha)=0.067587;th'(alpha)=-0.220586Iteration 770 complete. Error: 0.06758652747153975ARMIJO: th(0)=0.067650;th'(0)=-0.016287;	0.0 - 2.2202579190449872E-13 - Infinity	th(alpha)=0.067650 > 0.067650;th'(alpha)=-0.016604 >= -0.016286ARMIJO: th(0)=0.067650;th'(0)=-0.016287;	0.0 - 1.1101289595224936E-13 - 2.2202579190449872E-13	th(alpha)=0.067650 > 0.067650;th'(alpha)=-0.016287 >= -0.016286WOLFE: th(0)=0.067650;th'(0)=-0.016287;	0.0 - 3.7004298650749785E-14 - 1.1101289595224936E-13	th(alpha)=0.067650 <= 0.067650;th'(alpha)=-0.016808 < -0.016286ARMIJO: th(0)=0.067650;th'(0)=-0.016287;	3.7004298650749785E-14 - 7.400859730149957E-14 - 1.1101289595224936E-13	th(alpha)=0.067650 > 0.067650;th'(alpha)=-0.015786 >= -0.016286WOLFE: th(0)=0.067650;th'(0)=-0.016287;	3.7004298650749785E-14 - 5.550644797612468E-14 - 7.400859730149957E-14	th(alpha)=0.067650 <= 0.067650;th'(alpha)=-0.016473 < -0.016286ARMIJO: th(0)=0.067650;th'(0)=-0.016287;	5.550644797612468E-14 - 6.475752263881212E-14 - 7.400859730149957E-14	th(alpha)=0.067650 > 0.067650;th'(alpha)=-0.016679 >= -0.016286ARMIJO: th(0)=0.067650;th'(0)=-0.016287;	5.550644797612468E-14 - 6.01319853074684E-14 - 6.475752263881212E-14	th(alpha)=0.067650 > 0.067650;th'(alpha)=-0.016507 >= -0.016286mu >= nu: th(0)=0.067650;th'(0)=-0.016287;	5.550644797612468E-14 - 5.781921664179655E-14 - 6.01319853074684E-14Iteration 771 failed, retrying. Error: 0.06764979551135065Orientation vanished. Popping history element from 8Orientation vanished. Popping history element from 7Orientation vanished. Popping history element from 6Orientation vanished. Popping history element from 5Orientation vanished. Popping history element from 4ARMIJO: th(0)=0.075520;th'(0)=-21846604.446354;	0.0 - 1.3872480100509904E-6 - Infinity	th(alpha)=0.081215 > 0.072490;th'(alpha)=107.569742 >= -19661944.001719END: th(0)=0.075520;th'(0)=-21846604.446354;	0.0 - 6.936240050254952E-7 - 1.3872480100509904E-6	th(alpha)=0.069808;th'(alpha)=-2.493557Iteration 772 complete. Error: 0.069808472765988Orientation vanished. Popping history element from 4ARMIJO: th(0)=0.069781;th'(0)=-10158253.202887;	0.0 - 1.4943676182657766E-6 - Infinity	th(alpha)=0.073110 > 0.068263;th'(alpha)=35.235427 >= -9142427.882598END: th(0)=0.069781;th'(0)=-10158253.202887;	0.0 - 7.471838091328883E-7 - 1.4943676182657766E-6	th(alpha)=0.067631;th'(alpha)=1.118247Iteration 773 complete. Error: 0.0676308900960497Orientation vanished. Popping history element from 4ARMIJO: th(0)=0.067604;th'(0)=-11036279.767328;	0.0 - 1.6097587182260564E-6 - Infinity	th(alpha)=0.076539 > 0.065828;th'(alpha)=66.209304 >= -9932651.790595ARMIJO: th(0)=0.067604;th'(0)=-11036279.767328;	0.0 - 8.048793591130282E-7 - 1.6097587182260564E-6	th(alpha)=0.067511 > 0.066716;th'(alpha)=5.123023 >= -9932651.790595END: th(0)=0.067604;th'(0)=-11036279.767328;	0.0 - 2.6829311970434274E-7 - 8.048793591130282E-7	th(alpha)=0.066782;th'(alpha)=-0.365924Iteration 774 complete. Error: 0.06678150549262711Orientation vanished. Popping history element from 4END: th(0)=0.066756;th'(0)=-4071072.258912;	0.0 - 5.780200041879127E-7 - Infinity	th(alpha)=0.066424;th'(alpha)=1.042333Iteration 775 complete. Error: 0.06642443079710823Orientation vanished. Popping history element from 4ARMIJO: th(0)=0.066833;th'(0)=-6070196.720491;	0.0 - 1.245306348554814E-6 - Infinity	th(alpha)=0.067314 > 0.066077;th'(alpha)=5.470303 >= -5463177.048442END: th(0)=0.066833;th'(0)=-6070196.720491;	0.0 - 6.22653174277407E-7 - 1.245306348554814E-6	th(alpha)=0.066251;th'(alpha)=0.848448Iteration 776 complete. Error: 0.06625119234493002END: th(0)=0.066226;th'(0)=-1.972776;	0.0 - 1.2456772608355527E-13 - Infinity	th(alpha)=0.066226;th'(alpha)=-1.963524Iteration 777 complete. Error: 0.06622557970465504END: th(0)=0.066676;th'(0)=-10.395231;	0.0 - 2.68373030332801E-13 - Infinity	th(alpha)=0.066676;th'(alpha)=0.000000Iteration 778 complete. Error: 0.06667581347211238END: th(0)=0.066651;th'(0)=-1.816712;	0.0 - 5.781921664179655E-13 - Infinity	th(alpha)=0.066651;th'(alpha)=-1.812934Iteration 779 complete. Error: 0.0666508525711671END: th(0)=0.066625;th'(0)=-7.953777;	0.0 - 1.245677260835553E-12 - Infinity	th(alpha)=0.066625;th'(alpha)=0.000000Iteration 780 complete. Error: 0.0666250628407443END: th(0)=0.066599;th'(0)=-10.018458;	0.0 - 2.6837303033280106E-12 - Infinity	th(alpha)=0.066599;th'(alpha)=-9.737175Iteration 781 complete. Error: 0.0665993036105641WOLFE: th(0)=0.066594;th'(0)=-0.007696;	0.0 - 5.781921664179656E-12 - Infinity	th(alpha)=0.066594 <= 0.066594;th'(alpha)=-0.007725 < -0.007696END: th(0)=0.066594;th'(0)=-0.007696;	5.781921664179656E-12 - 1.1563843328359313E-11 - Infinity	th(alpha)=0.066594;th'(alpha)=-0.007653Iteration 782 complete. Error: 0.06659383813451972END: th(0)=0.066568;th'(0)=-0.000167;	0.0 - 2.4913545216711064E-11 - Infinity	th(alpha)=0.066568;th'(alpha)=-0.000120Iteration 783 failed, retrying. Error: 0.06656812199864098END: th(0)=0.066534;th'(0)=-0.000050;	0.0 - 5.367460606656022E-11 - Infinity	th(alpha)=0.066534;th'(alpha)=-0.000050Iteration 784 failed, retrying. Error: 0.06653386744127603END: th(0)=0.066483;th'(0)=-0.000053;	0.0 - 1.1563843328359314E-10 - Infinity	th(alpha)=0.066483;th'(alpha)=-0.000024Iteration 785 complete. Error: 0.0664826131319029END: th(0)=0.066457;th'(0)=-0.000000;	0.0 - 2.4913545216711066E-10 - Infinity	th(alpha)=0.066457;th'(alpha)=-0.000000Iteration 786 failed, retrying. Error: 0.06645702796581232WOLFE: th(0)=0.066409;th'(0)=-0.000000;	0.0 - 5.367460606656023E-10 - Infinity	th(alpha)=0.066409 <= 0.066409;th'(alpha)=-0.000000 < -0.000000END: th(0)=0.066409;th'(0)=-0.000000;	5.367460606656023E-10 - 1.0734921213312046E-9 - Infinity	th(alpha)=0.066409;th'(alpha)=-0.000000Iteration 787 failed, retrying. Error: 0.06640932607513478WOLFE: th(0)=0.066358;th'(0)=-0.000000;	0.0 - 2.312768665671863E-9 - Infinity	th(alpha)=0.066358 <= 0.066358;th'(alpha)=-0.000000 < -0.000000WOLFE: th(0)=0.066358;th'(0)=-0.000000;	2.312768665671863E-9 - 4.625537331343726E-9 - Infinity	th(alpha)=0.066358 <= 0.066358;th'(alpha)=-0.000000 < -0.000000WOLFE: th(0)=0.066358;th'(0)=-0.000000;	4.625537331343726E-9 - 1.3876611994031178E-8 - Infinity	th(alpha)=0.066358 <= 0.066358;th'(alpha)=-0.000000 < -0.000000WOLFE: th(0)=0.066358;th'(0)=-0.000000;	1.3876611994031178E-8 - 5.5506447976124713E-8 - Infinity	th(alpha)=0.066358 <= 0.066358;th'(alpha)=-0.000000 < -0.000000WOLFE: th(0)=0.066358;th'(0)=-0.000000;	5.5506447976124713E-8 - 2.7753223988062357E-7 - Infinity	th(alpha)=0.066358 <= 0.066358;th'(alpha)=-0.000000 < -0.000000WOLFE: th(0)=0.066358;th'(0)=-0.000000;	2.7753223988062357E-7 - 1.6651934392837414E-6 - Infinity	th(alpha)=0.066358 <= 0.066358;th'(alpha)=-0.000000 < -0.000000WOLFE: th(0)=0.066358;th'(0)=-0.000000;	1.6651934392837414E-6 - 1.165635407498619E-5 - Infinity	th(alpha)=0.066358 <= 0.066358;th'(alpha)=-0.000000 < -0.000000END: th(0)=0.066358;th'(0)=-0.000000;	1.165635407498619E-5 - 9.325083259988952E-5 - Infinity	th(alpha)=0.066358;th'(alpha)=-0.000000Iteration 788 complete. Error: 0.06635834716282406
```

If we test our model against the entire validation dataset, we get this accuracy:

Code from [MNistDemo.java:61](../.././src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L61) executed in 0.49 seconds: 
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
    85.76
```



Let's examine some incorrectly predicted results in more detail:

Code from [MNistDemo.java:75](../.././src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L75) executed in 0.60 seconds: 
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
![[5]](etc/bellsAndWhistles.1.png)   | 2 (98.9%), 4 (1.1%), 6 (0.0%)  
![[4]](etc/bellsAndWhistles.2.png)   | 6 (98.2%), 5 (1.8%), 0 (0.0%)  
![[6]](etc/bellsAndWhistles.3.png)   | 2 (100.0%), 6 (0.0%), 4 (0.0%) 
![[3]](etc/bellsAndWhistles.4.png)   | 2 (99.9%), 3 (0.1%), 9 (0.0%)  
![[7]](etc/bellsAndWhistles.5.png)   | 2 (99.6%), 7 (0.4%), 3 (0.0%)  
![[6]](etc/bellsAndWhistles.6.png)   | 2 (100.0%), 7 (0.0%), 6 (0.0%) 
![[9]](etc/bellsAndWhistles.7.png)   | 7 (96.6%), 9 (3.4%), 8 (0.0%)  
![[2]](etc/bellsAndWhistles.8.png)   | 9 (58.3%), 7 (21.1%), 2 (20.5%)
![[7]](etc/bellsAndWhistles.9.png)   | 9 (100.0%), 7 (0.0%), 5 (0.0%) 
![[4]](etc/bellsAndWhistles.10.png)  | 9 (80.2%), 4 (19.8%), 6 (0.0%) 
![[2]](etc/bellsAndWhistles.11.png)  | 9 (77.9%), 2 (22.0%), 8 (0.1%) 
![[4]](etc/bellsAndWhistles.12.png)  | 6 (93.1%), 4 (6.9%), 7 (0.0%)  
![[7]](etc/bellsAndWhistles.13.png)  | 3 (86.0%), 7 (14.0%), 9 (0.0%) 
![[7]](etc/bellsAndWhistles.14.png)  | 4 (99.7%), 9 (0.3%), 7 (0.0%)  
![[0]](etc/bellsAndWhistles.15.png)  | 3 (81.0%), 0 (18.8%), 2 (0.1%) 
![[7]](etc/bellsAndWhistles.16.png)  | 2 (99.0%), 7 (1.0%), 3 (0.0%)  
![[2]](etc/bellsAndWhistles.17.png)  | 8 (98.2%), 4 (1.3%), 9 (0.5%)  
![[9]](etc/bellsAndWhistles.18.png)  | 4 (86.8%), 9 (13.2%), 2 (0.0%) 
![[9]](etc/bellsAndWhistles.19.png)  | 2 (100.0%), 9 (0.0%), 3 (0.0%) 
![[5]](etc/bellsAndWhistles.20.png)  | 4 (100.0%), 5 (0.0%), 3 (0.0%) 
![[2]](etc/bellsAndWhistles.21.png)  | 3 (99.7%), 2 (0.3%), 0 (0.0%)  
![[9]](etc/bellsAndWhistles.22.png)  | 4 (100.0%), 9 (0.0%), 3 (0.0%) 
![[5]](etc/bellsAndWhistles.23.png)  | 7 (92.6%), 5 (7.4%), 0 (0.0%)  
![[6]](etc/bellsAndWhistles.24.png)  | 5 (100.0%), 8 (0.0%), 4 (0.0%) 
![[5]](etc/bellsAndWhistles.25.png)  | 7 (68.0%), 5 (32.0%), 8 (0.0%) 
![[7]](etc/bellsAndWhistles.26.png)  | 9 (85.2%), 7 (14.8%), 5 (0.0%) 
![[8]](etc/bellsAndWhistles.27.png)  | 7 (100.0%), 9 (0.0%), 8 (0.0%) 
![[5]](etc/bellsAndWhistles.28.png)  | 8 (91.5%), 5 (8.5%), 3 (0.0%)  
![[9]](etc/bellsAndWhistles.29.png)  | 5 (99.3%), 2 (0.7%), 9 (0.0%)  
![[7]](etc/bellsAndWhistles.30.png)  | 3 (99.9%), 4 (0.1%), 7 (0.0%)  
![[2]](etc/bellsAndWhistles.31.png)  | 9 (99.8%), 2 (0.2%), 7 (0.0%)  
![[3]](etc/bellsAndWhistles.32.png)  | 5 (100.0%), 3 (0.0%), 2 (0.0%) 
![[2]](etc/bellsAndWhistles.33.png)  | 1 (54.5%), 2 (45.5%), 3 (0.0%) 
![[9]](etc/bellsAndWhistles.34.png)  | 3 (100.0%), 9 (0.0%), 4 (0.0%) 
![[8]](etc/bellsAndWhistles.35.png)  | 4 (65.2%), 5 (34.7%), 8 (0.1%) 
![[3]](etc/bellsAndWhistles.36.png)  | 5 (80.9%), 3 (19.1%), 8 (0.0%) 
![[4]](etc/bellsAndWhistles.37.png)  | 9 (71.8%), 4 (28.2%), 2 (0.0%) 
![[5]](etc/bellsAndWhistles.38.png)  | 6 (98.5%), 4 (1.5%), 5 (0.0%)  
![[8]](etc/bellsAndWhistles.39.png)  | 4 (100.0%), 1 (0.0%), 3 (0.0%) 
![[4]](etc/bellsAndWhistles.40.png)  | 6 (100.0%), 4 (0.0%), 2 (0.0%) 
![[7]](etc/bellsAndWhistles.41.png)  | 2 (72.3%), 7 (27.7%), 3 (0.0%) 
![[3]](etc/bellsAndWhistles.42.png)  | 5 (100.0%), 4 (0.0%), 3 (0.0%) 
![[5]](etc/bellsAndWhistles.43.png)  | 1 (65.2%), 4 (34.7%), 5 (0.1%) 
![[9]](etc/bellsAndWhistles.44.png)  | 7 (100.0%), 8 (0.0%), 1 (0.0%) 
![[2]](etc/bellsAndWhistles.45.png)  | 7 (100.0%), 8 (0.0%), 9 (0.0%) 
![[0]](etc/bellsAndWhistles.46.png)  | 5 (74.4%), 0 (23.9%), 9 (1.7%) 
![[4]](etc/bellsAndWhistles.47.png)  | 9 (99.6%), 4 (0.4%), 8 (0.0%)  
![[9]](etc/bellsAndWhistles.48.png)  | 4 (88.1%), 9 (11.9%), 2 (0.0%) 
![[8]](etc/bellsAndWhistles.49.png)  | 3 (47.7%), 5 (44.7%), 0 (5.6%) 
![[6]](etc/bellsAndWhistles.50.png)  | 4 (59.3%), 6 (40.7%), 2 (0.1%) 
![[3]](etc/bellsAndWhistles.51.png)  | 7 (95.4%), 3 (4.6%), 9 (0.0%)  
![[5]](etc/bellsAndWhistles.52.png)  | 3 (100.0%), 0 (0.0%), 5 (0.0%) 
![[1]](etc/bellsAndWhistles.53.png)  | 7 (48.3%), 1 (38.7%), 3 (11.2%)
![[7]](etc/bellsAndWhistles.54.png)  | 9 (100.0%), 3 (0.0%), 1 (0.0%) 
![[9]](etc/bellsAndWhistles.55.png)  | 4 (100.0%), 9 (0.0%), 8 (0.0%) 
![[2]](etc/bellsAndWhistles.56.png)  | 7 (99.3%), 2 (0.6%), 3 (0.0%)  
![[5]](etc/bellsAndWhistles.57.png)  | 8 (85.8%), 5 (14.2%), 2 (0.0%) 
![[4]](etc/bellsAndWhistles.58.png)  | 9 (54.5%), 4 (45.5%), 5 (0.0%) 
![[3]](etc/bellsAndWhistles.59.png)  | 7 (100.0%), 3 (0.0%), 2 (0.0%) 
![[6]](etc/bellsAndWhistles.60.png)  | 5 (61.1%), 9 (38.9%), 6 (0.0%) 
![[9]](etc/bellsAndWhistles.61.png)  | 7 (82.9%), 9 (16.0%), 4 (1.0%) 
![[8]](etc/bellsAndWhistles.62.png)  | 5 (89.9%), 8 (10.1%), 2 (0.0%) 
![[5]](etc/bellsAndWhistles.63.png)  | 8 (75.2%), 5 (24.6%), 3 (0.2%) 
![[8]](etc/bellsAndWhistles.64.png)  | 9 (100.0%), 5 (0.0%), 2 (0.0%) 
![[9]](etc/bellsAndWhistles.65.png)  | 7 (99.2%), 9 (0.8%), 4 (0.0%)  
![[8]](etc/bellsAndWhistles.66.png)  | 4 (98.6%), 2 (1.4%), 8 (0.0%)  
![[8]](etc/bellsAndWhistles.67.png)  | 7 (100.0%), 9 (0.0%), 8 (0.0%) 
![[8]](etc/bellsAndWhistles.68.png)  | 5 (97.4%), 8 (2.6%), 7 (0.0%)  
![[2]](etc/bellsAndWhistles.69.png)  | 8 (100.0%), 2 (0.0%), 5 (0.0%) 
![[6]](etc/bellsAndWhistles.70.png)  | 0 (98.1%), 7 (1.9%), 6 (0.0%)  
![[9]](etc/bellsAndWhistles.71.png)  | 5 (99.9%), 8 (0.1%), 3 (0.0%)  
![[3]](etc/bellsAndWhistles.72.png)  | 5 (100.0%), 3 (0.0%), 2 (0.0%) 
![[3]](etc/bellsAndWhistles.73.png)  | 2 (75.1%), 3 (24.9%), 5 (0.0%) 
![[2]](etc/bellsAndWhistles.74.png)  | 8 (100.0%), 2 (0.0%), 3 (0.0%) 
![[6]](etc/bellsAndWhistles.75.png)  | 5 (95.3%), 6 (4.7%), 8 (0.0%)  
![[3]](etc/bellsAndWhistles.76.png)  | 7 (100.0%), 3 (0.0%), 8 (0.0%) 
![[8]](etc/bellsAndWhistles.77.png)  | 9 (72.3%), 8 (16.3%), 7 (11.3%)
![[7]](etc/bellsAndWhistles.78.png)  | 2 (100.0%), 7 (0.0%), 9 (0.0%) 
![[5]](etc/bellsAndWhistles.79.png)  | 8 (98.6%), 5 (1.4%), 3 (0.0%)  
![[5]](etc/bellsAndWhistles.80.png)  | 8 (68.4%), 2 (31.4%), 4 (0.1%) 
![[9]](etc/bellsAndWhistles.81.png)  | 5 (99.1%), 4 (0.8%), 3 (0.1%)  
![[5]](etc/bellsAndWhistles.82.png)  | 7 (100.0%), 8 (0.0%), 0 (0.0%) 
![[9]](etc/bellsAndWhistles.83.png)  | 7 (51.5%), 9 (48.5%), 4 (0.0%) 
![[8]](etc/bellsAndWhistles.84.png)  | 5 (54.0%), 2 (44.0%), 8 (2.1%) 
![[5]](etc/bellsAndWhistles.85.png)  | 3 (91.8%), 5 (7.6%), 1 (0.6%)  
![[3]](etc/bellsAndWhistles.86.png)  | 5 (100.0%), 4 (0.0%), 9 (0.0%) 
![[6]](etc/bellsAndWhistles.87.png)  | 5 (75.9%), 8 (14.1%), 6 (10.0%)
![[7]](etc/bellsAndWhistles.88.png)  | 2 (100.0%), 7 (0.0%), 9 (0.0%) 
![[4]](etc/bellsAndWhistles.89.png)  | 8 (99.5%), 1 (0.3%), 7 (0.1%)  
![[6]](etc/bellsAndWhistles.90.png)  | 2 (99.9%), 6 (0.0%), 7 (0.0%)  
![[3]](etc/bellsAndWhistles.91.png)  | 8 (100.0%), 2 (0.0%), 3 (0.0%) 
![[3]](etc/bellsAndWhistles.92.png)  | 5 (100.0%), 3 (0.0%), 4 (0.0%) 
![[3]](etc/bellsAndWhistles.93.png)  | 2 (99.9%), 3 (0.1%), 8 (0.0%)  
![[9]](etc/bellsAndWhistles.94.png)  | 4 (99.5%), 9 (0.5%), 2 (0.0%)  
![[3]](etc/bellsAndWhistles.95.png)  | 6 (100.0%), 2 (0.0%), 3 (0.0%) 
![[2]](etc/bellsAndWhistles.96.png)  | 1 (66.5%), 2 (33.5%), 3 (0.0%) 
![[8]](etc/bellsAndWhistles.97.png)  | 3 (94.9%), 4 (5.1%), 2 (0.0%)  
![[7]](etc/bellsAndWhistles.98.png)  | 9 (100.0%), 7 (0.0%), 3 (0.0%) 
![[7]](etc/bellsAndWhistles.99.png)  | 3 (96.8%), 9 (1.2%), 4 (1.1%)  
![[0]](etc/bellsAndWhistles.100.png) | 9 (100.0%), 5 (0.0%), 0 (0.0%) 




