import React, {Fragment} from "react";

const AbstactBlock = () => (
    <section className="section">
        <div className="container is-max-desktop">
            <div className="columns is-centered has-text-centered">
                <div className="column is-four-fifths">
                    <h2 className="title is-3">Abstract</h2>
                    <div className="content has-text-justified">
			<p style={{fontStyle: "italic"}}>
			     Self-supervised learning for inverse problems allows to train a reconstruction network from noise and/or incomplete data alone. These methods have the potential of enabling learning-based solutions when obtaining ground-truth references for training is expensive or even impossible. In this paper, we propose a new self-supervised learning strategy devised for the challenging setting where measurements are observed via a single incomplete observation model. We introduce a new definition of equivariance in the context of reconstruction networks, and show that the combination of self-supervised splitting losses and equivariant reconstruction networks results in the same minimizer in expectation as the one of a supervised loss. Through a series of experiments on image inpainting, accelerated magnetic resonance imaging, and compressive sensing, we demonstrate that the proposed loss achieves state-of-the-art performance in settings with highly rank-deficient forward models.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </section>
)

const Section1 = () => {
return (
    <Fragment>
        <br />
        <AbstactBlock />
    </Fragment>
);
}

export default Section1;
