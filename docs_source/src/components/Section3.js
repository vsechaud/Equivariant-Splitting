import React, { useState } from "react";
import { ToggleButton, ToggleButtonGroup } from '@mui/material';
import CompressiveSensingChart from './CompressiveSensingChart';

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

// Table component for results
const ResultsTable = ({ title, headers, rows, caption }) => {
    return (
        <div style={{ margin: '2rem 0', overflowX: 'auto' }}>
            <h3 className="title is-4">{title}</h3>
            <table style={{
                width: '100%',
                borderCollapse: 'collapse',
                margin: '1rem auto',
                maxWidth: '800px'
            }}>
                <thead>
                    <tr style={{ borderBottom: '2px solid #333' }}>
                        {headers.map((header, idx) => (
                            <th key={idx} style={{
                                padding: '0.75rem',
                                textAlign: 'center',
                                fontWeight: '600'
                            }}>
                                {header}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row, rowIdx) => (
                        <tr key={rowIdx} style={{
                            borderBottom: rowIdx === rows.length - 1 ? '2px solid #333' : '1px solid #ddd',
                            backgroundColor: row.isSeparator ? '#f5f5f5' : 'transparent'
                        }}>
                            {row.cells.map((cell, cellIdx) => (
                                <td key={cellIdx} style={{
                                    padding: '0.75rem',
                                    fontWeight: cell.bold ? '600' : 'normal'
                                }}>
                                    {cell.value}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
            {caption && (
                <p style={{
                    fontSize: '0.9rem',
                    color: '#666',
                    marginTop: '0.5rem',
                    fontStyle: 'italic'
                }}>
                    {caption}
                </p>
            )}
        </div>
    );
};

// Sample reconstructions component
const SampleReconstructions = () => {
    const [task, setTask] = useState('inpainting');

    const taskData = {
        inpainting: {
            title: 'Image Inpainting',
            images: [
                { name: 'Ground Truth', file: 'Inpainting_GT.webp' },
                { name: 'Supervised', file: 'Inpainting_EQ_Supervised.webp' },
                { name: 'ES (Ours)', file: 'Inpainting_EQ_ES.webp' },
                { name: 'EI', file: 'Inpainting_EQ_EI.webp' },
                { name: 'Measurement', file: 'Inpainting_Measurement.webp' },
            ],
            caption: 'ES produces images closer to the supervised baseline than EI which appears blurry.'
        },
        mri: {
            title: 'MRI Reconstruction (×8 Accel., 40 dB SNR)',
            images: [
                { name: 'Ground Truth', file: 'MRIx8_GT.webp' },
                { name: 'Supervised', file: 'MRIx8_EQ_Supervised.webp' },
                { name: 'ES (Ours)', file: 'MRIx8_EQ_ES.webp' },
                { name: 'EI', file: 'MRIx8_EQ_EI.webp' },
                { name: 'SURE', file: 'MRIx8_EQ_SURE.webp' },
                { name: 'IDFT', file: 'MRIx8_IDFT.webp' },
            ],
            caption: 'ES is closer to supervised baseline. EI suffers from artifacts, while SURE and IDFT fail to recover unobserved frequencies.'
        }
    };

    const currentData = taskData[task];

    return (
        <div style={{ margin: '2rem 0' }}>
            <h3 className="title is-4">Visual comparisons</h3>
            <ToggleButtonGroup
                color="primary"
                value={task}
                exclusive
                onChange={(e, newTask) => newTask && setTask(newTask)}
                aria-label="Task selection"
                style={{ marginBottom: '1.5rem' }}
            >
                <ToggleButton value="inpainting">Inpainting</ToggleButton>
                <ToggleButton value="mri">MRI</ToggleButton>
            </ToggleButtonGroup>

            <div style={{
                display: 'flex',
                flexWrap: 'wrap',
                justifyContent: 'center',
                gap: '1rem',
                marginTop: '1rem'
            }}>
                {currentData.images.map((img, idx) => (
                    <div key={idx} style={{ textAlign: 'center', maxWidth: '150px' }}>
                        <img
                            src={process.env.PUBLIC_URL + '/imgs/figures/' + img.file}
                            alt={img.name}
                            style={{ width: '100%', height: 'auto', borderRadius: '4px' }}
                        />
                        <p style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>{img.name}</p>
                    </div>
                ))}
            </div>
            <p style={{
                fontSize: '0.9rem',
                color: '#666',
                marginTop: '1rem',
                fontStyle: 'italic'
            }}>
                {currentData.caption}
            </p>
        </div>
    );
};

const Content = () => {
    // MRI Results Table
    const mriTable = {
        title: 'Performance for MRI',
        headers: ['Method', 'PSNR ↑', 'SSIM ↑', 'EQUIV ↑'],
        rows: [
            { cells: [
                { value: 'Supervised' },
                { value: '28.74 ± 2.81' },
                { value: '0.6445 ± 0.1094' },
                { value: '31.71 ± 2.83' }
            ], isSeparator: false },
            { cells: [
                { value: 'ES (Ours)', bold: true },
                { value: '28.54 ± 2.75', bold: true },
                { value: '0.6195 ± 0.1188', bold: true },
                { value: '31.53 ± 2.74', bold: true }
            ], isSeparator: false },
            { cells: [
                { value: 'EI' },
                { value: '27.88 ± 2.64' },
                { value: '0.5731 ± 0.1299' },
                { value: '30.79 ± 2.64' }
            ], isSeparator: false },
            { cells: [
                { value: 'SURE' },
                { value: '24.45 ± 1.86' },
                { value: '0.5479 ± 0.0740' },
                { value: '27.35 ± 1.90' }
            ], isSeparator: false },
            { cells: [
                { value: 'IDFT' },
                { value: '23.62 ± 1.90' },
                { value: '0.5052 ± 0.0900' },
                { value: '25.99 ± 1.94' }
            ], isSeparator: false }
        ],
        caption: 'ES performs better than EI and SURE while nearly matching supervised performance. Values: avg ± std.'
    };

    // Inpainting Results Table
    const inpaintingTable = {
        title: 'Performance for Inpainting',
        headers: ['Method', 'PSNR ↑', 'SSIM ↑', 'EQUIV ↑'],
        rows: [
            { cells: [
                { value: 'Supervised' },
                { value: '28.46 ± 2.97' },
                { value: '0.8982 ± 0.0411' },
                { value: '28.46 ± 2.97' }
            ], isSeparator: false },
            { cells: [
                { value: 'ES (Ours)', bold: true },
                { value: '27.45 ± 2.86', bold: true },
                { value: '0.8737 ± 0.0461', bold: true },
                { value: '27.46 ± 2.85', bold: true }
            ], isSeparator: false },
            { cells: [
                { value: 'EI' },
                { value: '25.89 ± 2.65' },
                { value: '0.8332 ± 0.0521' },
                { value: '25.89 ± 2.65' }
            ], isSeparator: false },
            { cells: [
                { value: 'MC' },
                { value: '8.22 ± 2.47' },
                { value: '0.0983 ± 0.0551' },
                { value: '8.22 ± 2.47' }
            ], isSeparator: false },
            { cells: [
                { value: 'Incomplete image' },
                { value: '8.22 ± 2.47' },
                { value: '0.0973 ± 0.0542' },
                { value: 'N/A' }
            ], isSeparator: false }
        ],
        caption: 'ES outperforms EI and performs competitively against supervised baseline. Values: avg ± std.'
    };

    // Ablation Study Table
    const ablationTable = {
        title: 'Impact of Equivariant Architectures',
        headers: ['Training Loss', 'Eq. Arch.', 'Task', 'PSNR ↑', 'SSIM ↑', 'EQUIV ↑'],
        rows: [
            { cells: [
                { value: 'Supervised' },
                { value: '✓' },
                { value: 'Inpainting' },
                { value: '28.46 ± 2.97' },
                { value: '0.8982 ± 0.0411' },
                { value: '28.46 ± 2.97' }
            ], isSeparator: false },
            { cells: [
                { value: 'Supervised' },
                { value: '✗' },
                { value: 'Inpainting' },
                { value: '28.62 ± 3.03' },
                { value: '0.9002 ± 0.0414' },
                { value: '27.85 ± 2.71' }
            ], isSeparator: false },
            { cells: [
                { value: 'Splitting (Ours)', bold: true },
                { value: '✓', bold: true },
                { value: 'Inpainting' },
                { value: '27.45 ± 2.86', bold: true },
                { value: '0.8737 ± 0.0461', bold: true },
                { value: '27.46 ± 2.85', bold: true }
            ], isSeparator: false },
            { cells: [
                { value: 'Splitting' },
                { value: '✗' },
                { value: 'Inpainting' },
                { value: '27.20 ± 2.83' },
                { value: '0.8652 ± 0.0461' },
                { value: '26.52 ± 2.60' }
            ], isSeparator: false },
            { cells: [
                { value: 'Supervised' },
                { value: '✓' },
                { value: 'MRI' },
                { value: '28.74 ± 2.81' },
                { value: '0.6445 ± 0.1094' },
                { value: '31.71 ± 2.83' }
            ], isSeparator: false },
            { cells: [
                { value: 'Supervised' },
                { value: '✗' },
                { value: 'MRI' },
                { value: '28.48 ± 2.68' },
                { value: '0.6381 ± 0.1082' },
                { value: '28.78 ± 1.95' }
            ], isSeparator: false },
            { cells: [
                { value: 'Splitting (Ours)', bold: true },
                { value: '✓', bold: true },
                { value: 'MRI' },
                { value: '28.54 ± 2.75', bold: true },
                { value: '0.6195 ± 0.1188', bold: true },
                { value: '31.53 ± 2.74', bold: true }
            ], isSeparator: false },
            { cells: [
                { value: 'Splitting' },
                { value: '✗' },
                { value: 'MRI' },
                { value: '28.18 ± 2.58' },
                { value: '0.6104 ± 0.1176' },
                { value: '27.28 ± 2.10' }
            ], isSeparator: false }
        ],
        caption: 'Equivariant architectures synergize with splitting loss for improved performance.'
    };

    return (
        <div>
            <CompressiveSensingChart />
            <ResultsTable {...mriTable} />
            <ResultsTable {...inpaintingTable} />
            <SampleReconstructions />
            <ResultsTable {...ablationTable} />
        </div>
    );
};

const Section3 = () => {
    return (
        <CenterWrapper content={<Content />} />
    );
};

export default Section3;
