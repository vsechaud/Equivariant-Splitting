import React, { useState } from "react";
import { Grid, ToggleButton, ToggleButtonGroup } from '@mui/material';
import ReactSwipe from 'react-swipe'
import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';

const CenterWrapper = (props) => {
    return (
        <section className="section">
            <div className="container is-max-desktop">
                <div className="columns is-centered has-text-centered">
                    <div className="column is-four-fifths">
                        {props.content}
                    </div>
                </div>
            </div>
        </section>
    );
}

const ImageCompareSlider = ({imgs}) => {
    return (
	<div style={{width: "50%", margin: "auto"}}>
		<ReactCompareSlider
		    itemOne={<ReactCompareSliderImage src={imgs.input} alt='input image'/>}
		    itemTwo={<ReactCompareSliderImage src={imgs.recon} alt='recon image'/>}
		/>
	</div>
    );
}

const Carousel = ({images}) => {
    return (
            <Grid container direction="column" style={{margin: '1.5rem 0 0 0'}}>
                <Grid container direction="row">
                    <Grid item xs={12} md={12} sm={12}>
                        <ReactSwipe
                            className="carousel"
                            swipeOptions={{continuous: true, disableScroll: true}}
                            childCount={images.length}
                            >
                            {images.map((image_pair) => {
                                return (
                                    <div>
                                        <ImageCompareSlider imgs={image_pair}/>
                                    </div>
                                );
                                })}
                        </ReactSwipe>
                    </Grid>
                </Grid>
            </Grid>
    );
}



function range(start, end){
    let array = [];
    for (let i=start; i<end; i++){
        array.push(i);
    }
    return array;
}

const IamgeDisplay = ({task}) => {
    const images = range(0, 1).map((idx) => {
        return ({
            'input': process.env.PUBLIC_URL + '/imgs/results/' + task + '/input_'+idx+'.webp',
            'recon': process.env.PUBLIC_URL + '/imgs/results/' + task + '/recon_'+idx+'.webp',
        });
    })

    return (
        <Carousel images={images}/>
    )
}


const Content = () => {
    const task_pair = {'inpainting':'Image Inpainting',
                       'mri':'Accelerated MRI'}

    const tasks = ['inpainting','mri'];
    const [task, setTask] = useState('inpainting');

    const onTaskToggle = (button_val) => {
        setTask(button_val);
    };

    return (
        <div>
            <h2 className="title is-3">Reconstructions</h2>
            <ToggleButtonGroup
                    color="primary"
                    value={task}
                    aria-label="Platform">
                {tasks.map(t => (
                    <ToggleButton value={t} onClick={()=>{onTaskToggle(t)}} id={t} key={t}>
                    {task_pair[t]}
                    </ToggleButton>))
                }
            </ToggleButtonGroup>

            <IamgeDisplay task={task}/>
        </div>
    );
}

const Section3 = () => {
    return (
        <CenterWrapper content={<Content />}/>
    );
}

export default Section3
