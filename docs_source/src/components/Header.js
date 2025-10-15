import { Button } from "@mui/material";
import React, {Component} from "react";
import { VscGithub } from "react-icons/vsc"
import {SiArxiv} from "react-icons/si"
import {FaGlobe} from "react-icons/fa"

const AuthorBlock = (props) => (
    <span className="author-block">
        <a href={props.link}>{props.name}</a>
        <sup>{props.number}</sup>
    </span>
)

const LinkButton = (props) => (
    <Button sx={{m: '0.3rem'}}
            style={{
                borderRadius: 35,
                backgroundColor: "black",
                padding: "0.5rem 1.0rem"
            }}
            href={props.link}
            variant="contained"
            startIcon={props.icon}>
        {props.text}
    </Button>
);

export default class Header extends Component{
    render(){
        return (
            <section className="hero information">
                <div className="container is-max-desktop">
                    <div className="columns is-centered">
                        <div className="column has-text-centered">
                            <h1 className="title is-1 publication-title">
                                Equivariant Splitting: Self-supervised learning from incomplete data
                            </h1>
                            <div className="is-size-5 publication-authors">
                                <AuthorBlock name="Victor Sechaud" link="#" number=" * 1" />,&nbsp;
                                <AuthorBlock name="Jérémy Scanvic" link="#" number=" * 1, 2" />,&nbsp;
                                <AuthorBlock name="Quentin Barthélemy" link="#" number=" 2" />,&nbsp;
                                <AuthorBlock name="Patrice Abry" link="#" number=" 1" />,&nbsp;
                                <AuthorBlock name="Julián Tachella" link="#" number=" 1" />
                            </div>
                            <div className="is-size-5 publication-authors">
                                <span className="author-block"><sup>1</sup> LPENSL, CNRS, ENS de Lyon, France</span>
                                <span className="author-block">&nbsp;<sup>2</sup> Prysm, Lyon, France</span>
                                <br></br>
                                <span className="author-block-small">* Equal contribution</span>
                            </div>
                            {/*Publication links*/}
                            <div className="column has-text-centered">
                                <LinkButton link={"https://arxiv.org/abs/2510.00929"} icon={<SiArxiv />} text="arXiv"/>
                                <LinkButton link={"https://github.com/vsechaud/Equivariant-Splitting"} icon={<VscGithub />} text="Code"/>
                                <LinkButton link={"."} icon={<FaGlobe />} text="Project"/>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        );
    }
}
