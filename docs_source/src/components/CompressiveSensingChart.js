import React from 'react';

const CompressiveSensingChart = () => {
    return (
        <div style={{ margin: '2rem 0' }}>
            <h3 className="title is-4">Performance for Compressive Sensing</h3>
            <div style={{
                maxWidth: '900px',
                margin: '0 auto',
                padding: '1.5rem',
                backgroundColor: '#fafafa',
                borderRadius: '8px',
                display: 'flex',
                justifyContent: 'center'
            }}>
                <img
                    src={process.env.PUBLIC_URL + '/imgs/cs_chart.svg'}
                    alt="ES performs competitively against the supervised baseline while EI shows a gap increasing with the compression level"
                    style={{
                        width: '100%',
                        maxWidth: '800px',
                        height: 'auto'
                    }}
                />
            </div>
            <p style={{
                fontSize: '0.9rem',
                color: '#666',
                marginBottom: '1rem',
                fontStyle: 'italic'
            }}>
                ES performs competitively against the supervised baseline while EI shows a gap increasing with the compression level
            </p>
        </div>
    );
};

export default CompressiveSensingChart;
