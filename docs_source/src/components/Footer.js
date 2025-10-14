import React, {Component} from "react";
import { IconButton } from "@mui/material";
import { VscGithub } from "react-icons/vsc"
import {FaFilePdf} from "react-icons/fa"
import {FaGlobe} from "react-icons/fa"

const LinkButton = (props) => (
    <IconButton href={props.link}>
        {props.icon}
  </IconButton>
);

export default class Footer extends Component{
  render(){
      return (
        <footer className="footer">
        <div className="container">
            <div className="content has-text-centered">
            <LinkButton link={"https://arxiv.org/abs/2510.00929"} icon={<FaFilePdf />} text="Paper"/>
            <LinkButton link={"https://github.com/vsechaud/Equivariant-Splitting"} icon={<VscGithub />} text="Code"/>
	<LinkButton link={"."} icon={<FaGlobe />} text="Project"/>
            </div>
            <div className="columns is-centered">
            <div className="column is-8">
                <div className="content">
                <p>
                    This website is licensed under a <a rel="license"
                                                        href="http://creativecommons.org/licenses/by-sa/4.0/">Creative
                    Commons Attribution-ShareAlike 4.0 International License</a>.
                </p>
                <p>
                    This website is adapted from the <a
                    href="https://github.com/BlindDPS/blind-dps-project">Blind DPS project page</a>.
                </p>
                </div>
            </div>
            </div>
        </div>
        </footer>
    );
  }
}
