'use strict';
// Authored keys for the winder. Coordinates are figure-local (see winder.js).
const POSES = (() => {
  const p = Winder.pose;
  return {
    rest: p(),
    // holding a fallen tag up toward the board
    offer: p({ lean: .13, tilt: -.12, look: [.6, -.7], handB: [82, -186], handA: [-14, -80], footB: [19, 0], mouth: 'flat' }),
    // the first breath of wind lifts the scarf; the tag strains in the mitten
    strain: p({ lean: .03, tilt: -.14, look: [.3, -.9], eyes: 'wide', brow: .6, handB: [72, -192], handA: [-22, -84], footB: [19, 0],
      scarf: { angle: 2.75, wave: .4, phase: 1.2 }, hair: -.45, hem: -5, flare: 3 }),
    // the tag is gone: open hand, head back, eyes after it
    gone: p({ lean: -.08, tilt: -.3, look: [-.7, -1], eyes: 'wide', brow: .8, mouth: 'o', handB: [58, -168], handA: [-30, -92], footB: [19, 0],
      scarf: { angle: 2.3, wave: .3, phase: 2.6 }, hair: -.2, hem: -2 }),
    // looks back at the board: every tag is trembling
    uhoh: p({ lean: -.03, tilt: .02, look: [.8, -.35], eyes: 'wide', brow: 1, mouth: 'flat', handB: [34, -118], handA: [-24, -88], footB: [19, 0],
      scarf: { angle: 2.1, wave: .25, phase: 3.4 }, hem: 0 }),
    // the gust lands: shoved back on the heels
    hit: p({ pelvis: [-10, -82], lean: -.24, tilt: -.12, look: [.4, -.2], eyes: 'closed', brow: -1, mouth: 'o', handB: [40, -158], handA: [-58, -112], bendA: -1,
      footA: [-40, 0], footB: [20, 0], pitchB: -.35, scarf: { angle: 3.05, wave: .5, phase: 4.3, len: 84 }, hair: -1, hem: -14, flare: 9 }),
    // braced: weight low and forward, forearm across the face
    brace: p({ pelvis: [-4, -72], lean: .4, tilt: .1, look: [.5, .2], eyes: 'closed', brow: -1, mouth: 'flat', handB: [72, -164], handA: [-44, -70],
      footA: [-54, 0], pitchA: .12, footB: [24, 0], scarf: { angle: 3.2, wave: .55, phase: 0, len: 88 }, hair: -1, hem: -17, flare: 10 }),
    // the wind drops: arm lowers, one eye opens over it
    peek: p({ pelvis: [-3, -78], lean: .2, tilt: -.06, look: [.6, -.5], eyes: 'open', brow: .5, handB: [66, -132], handA: [-34, -74],
      footA: [-46, 0], footB: [22, 0], scarf: { angle: 2.2, wave: .3, phase: 1 }, hair: -.25, hem: -5, flare: 2 }),
    // the back foot comes in under the body: a real step, lifted
    stepIn: p({ pelvis: [-2, -80], lean: .04, tilt: -.2, look: [.6, -.9], eyes: 'wide', brow: .9, handB: [44, -104], handA: [-30, -78],
      footA: [-36, -9], pitchA: .35, footB: [20, 0] }),
    // upright, head back: the star
    wonder: p({ pelvis: [-2, -85], lean: -.07, tilt: -.26, look: [.55, -1], eyes: 'wide', brow: 1, mouth: 'o', handB: [30, -92], handA: [-26, -80],
      footA: [-24, 0], footB: [16, 0], scarf: { angle: 1.75, wave: .18, phase: 2 }, hair: .05 }),
    // winding: the spool hand pays out thread, up and down with each nail
    feedA: p({ lean: .06, tilt: -.06, look: [.6, -.5], handB: [64, -128], handA: [-10, -80], footB: [19, 0] }),
    feedB: p({ pelvis: [1, -86], lean: .1, tilt: -.1, look: [.6, -.6], handB: [72, -150], handA: [-14, -82], footB: [19, 0] }),
    tuck: p({ lean: .02, tilt: .22, look: [.2, .9], handB: [24, -104], handA: [-12, -80], footB: [19, 0] }),
    hmm: p({ lean: -.06, tilt: -.16, look: [.7, -.6], brow: -.5, mouth: 'flat', handB: [40, -150], handA: [-16, -78], footB: [19, 0] }),
    // flicking tags up onto the nails like cards
    windup: p({ pelvis: [-2, -80], lean: -.08, tilt: .06, look: [.5, -.3], handB: [4, -92], handA: [26, -98], bendA: -1, footB: [19, 0] }),
    flick: p({ pelvis: [1, -87], lean: .12, tilt: -.14, look: [.6, -.8], mouth: 'smile', handB: [70, -182], handA: [24, -100], bendA: -1, footB: [19, 0] }),
    proud: p({ lean: -.06, tilt: -.12, look: [.6, -.6], mouth: 'smile', handB: [22, -92], handA: [-24, -92], bendA: -1, footB: [19, 0] }),
    point: p({ lean: .12, tilt: -.06, look: [.8, -.3], eyes: 'wide', brow: .6, handB: [84, -160], handA: [-14, -80], footB: [19, 0] }),
    point2: p({ lean: .08, tilt: -.2, look: [.7, -.9], eyes: 'wide', brow: .8, mouth: 'flat', handB: [74, -190], handA: [-14, -80], footB: [19, 0] }),
    lookdown: p({ lean: .04, tilt: .3, look: [.1, 1], brow: .6, mouth: 'flat', handB: [34, -104], handA: [16, -100], bendA: -1, footB: [19, 0] }),
    shrug: p({ pelvis: [0, -86], lean: -.04, tilt: -.08, look: [.5, -.2], brow: 1, mouth: 'flat', handB: [56, -132], handA: [-56, -126], bendA: -1, footB: [19, 0] }),
    // the long night, sitting against nothing in particular
    doze: p({ pelvis: [0, -37], lean: .1, tilt: .44, look: [0, .6], eyes: 'closed', footA: [50, 0], footB: [66, 0], pitchA: -.9, pitchB: -.8, handA: [22, -44], handB: [30, -40],
      scarf: { angle: 1.7, wave: .12, phase: 1, len: 64 } }),
    doze2: p({ pelvis: [0, -36], lean: .13, tilt: .5, look: [0, .6], eyes: 'closed', footA: [50, 0], footB: [66, 0], pitchA: -.9, pitchB: -.8, handA: [22, -43], handB: [30, -39],
      scarf: { angle: 1.7, wave: .12, phase: 1, len: 64 } }),
    smile: p({ pelvis: [-2, -85], lean: -.05, tilt: -.22, look: [.55, -.9], eyes: 'open', brow: .6, mouth: 'smile', handB: [24, -84], handA: [-22, -78],
      footA: [-24, 0], footB: [16, 0], scarf: { angle: 1.82, wave: .2, phase: 2.4 } }),
  };
})();
