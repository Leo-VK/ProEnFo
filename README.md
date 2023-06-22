# ProEnFo

## Introduction

This is the code related to the paper 
"Benchmarks and Custom Package for Electrical Load Forecasting"(https://openreview.net/forum?id=O61RXF9dvD&invitationId=NeurIPS.cc/2023/Track/Datasets_and_Benchmarks/Submission433/-/Supplementary_Material&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FTrack%2FDatasets_and_Benchmarks%2FAuthors%23author-tasks)) submitted to Neurips 2023 Datasets and Benchmarks Track. 
This repository mainly aims at implementing routines for probabilistic energy forecasting. However, we also provide the implementation of relevant point forecasting models.
The datasets and their forecasting results in this archive can be found at https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3009646_connect_hku_hk/Euy4Rv8DsM1Cu1hJ85yHL18BNsDNbS5XiaVoCvl-l-07tQ?e=OFLF3A. 
To reproduce the results in our archive, users can refer to the process in the main.py file. By selecting different Feature engineering methods and preprocessing, post-processing, and training models, users can easily construct different forecasting models.

## Dataset
We include several different datasets in our load forecasting archive, here is the summary of them.
<table class=MsoTableGrid border=1 cellspacing=0 style="border-collapse:collapse;width:463.8500pt;mso-table-layout-alt:fixed;
        border:none;mso-border-left-alt:0.5000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;
        mso-border-right-alt:0.5000pt solid windowtext;mso-border-bottom-alt:0.5000pt solid windowtext;mso-border-insideh:0.5000pt solid windowtext;
        mso-border-insidev:0.5000pt solid windowtext;mso-padding-alt:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;">
            <tr style="height:20.7000pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-weight:normal;font-size:10.5000pt;
                        mso-font-kerning:1.0000pt;">
                            <font face="Calibri">Dataset</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-weight:normal;font-size:10.5000pt;
                        mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-weight:normal;font-size:10.5000pt;
                        mso-font-kerning:1.0000pt;">
                            <font face="Calibri">No. of serie</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-weight:normal;font-size:10.5000pt;
                        mso-font-kerning:1.0000pt;">
                            <font face="Calibri">s</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-weight:normal;font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-weight:normal;font-size:10.5000pt;
                        mso-font-kerning:1.0000pt;">
                            <font face="Calibri">Length</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-weight:normal;font-size:10.5000pt;
                        mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-weight:normal;font-size:10.5000pt;
                        mso-font-kerning:1.0000pt;">
                            <font face="Calibri">Resolution</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-weight:normal;font-size:10.5000pt;
                        mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-weight:normal;font-size:10.5000pt;
                        mso-font-kerning:1.0000pt;">
                            <font face="Calibri">Load</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-weight:normal;font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <span style="mso-spacerun:'yes';">&nbsp;</span>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-weight:normal;font-size:10.5000pt;
                        mso-font-kerning:1.0000pt;">
                            <font face="Calibri">type</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-weight:normal;font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:1.0000pt solid windowtext;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-weight:normal;font-size:10.5000pt;
                        mso-font-kerning:1.0000pt;">
                            <font face="Calibri">External variables</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-weight:normal;font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
            <tr style="height:20.7000pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">1</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">Covid</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">19</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">1</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">31912</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">hourly</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">aggregated</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">airTemperature</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">,</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">Humidity, etc</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
            <tr style="height:16.1500pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">2</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">GEF</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">12</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">20</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">39414</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">hourly</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">aggregated</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">airTemperature</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
            <tr style="height:16.1500pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">3</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">GEF</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">1</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">4</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">1</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">17520</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">hourly</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">aggregated</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">airTemperature</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
            <tr style="height:16.1500pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">4</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">GEF</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">1</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">7</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">8</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">17544</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">hourly</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">aggregated</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">airTemperature</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
            <tr style="height:16.1500pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">5</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">PDB</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">1</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">17520</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">hourly</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">aggregated</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">airTemperature</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
            <tr style="height:16.1500pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">6</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">Spanish</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">1</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">35064</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">hourly</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">aggregated</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">airTemperature</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">,</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">seaLvlPressure, etc</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
            <tr style="height:16.1500pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">7</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">Hog</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">24</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">17544</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">hourly</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">building</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">airTemperature</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">,</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">wind speed, etc.</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
            <tr style="height:16.1500pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">8</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">Bull</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">41</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">17544</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">hourly</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">building</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">airTemperature</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">,</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">wind speed, etc.</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
            <tr style="height:16.1500pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">9</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">Cockatoo</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">1</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">17544</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">hourly</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">building</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">airTemperature</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">,</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">wind speed, etc.</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
            <tr style="height:16.1500pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">10</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">ELF</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">1</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">21792</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">hourly</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">aggregated</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">No</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
            <tr style="height:16.6500pt;">
                <td width=33 valign=top style="width:33.6000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">11</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=80 valign=top style="width:80.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">UCI</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=69 valign=top style="width:69.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">321</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=46 valign=top style="width:46.1500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">26304</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=61 valign=top style="width:61.5000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">hourly</font>
                        </span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=67 valign=top style="width:67.7000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <font face="Calibri">building</font>
                        </span>
                        <span style="font-family:宋体;mso-ascii-font-family:Calibri;mso-hansi-font-family:Calibri;
                        mso-bidi-font-family:'Times New Roman';font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
                <td width=105 valign=top style="width:105.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:1.0000pt solid windowtext;
                mso-border-left-alt:0.5000pt solid windowtext;border-right:1.0000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;
                border-top:none;mso-border-top-alt:0.5000pt solid windowtext;border-bottom:1.0000pt solid windowtext;
                mso-border-bottom-alt:0.5000pt solid windowtext;">
                    <p class=MsoNormal align=center style="text-align:center;">
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">No</span>
                        <span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
                        font-size:10.5000pt;mso-font-kerning:1.0000pt;">
                            <o:p></o:p>
                        </span>
                    </p>
                </td>
            </tr>
        </table>


## Prerequisites
- Python 
- Conda

### Create a virtual environment
This is only needed when used the first time on the machine.

```bash
conda env create --file proenfo_env.yml
```

### Activate and deactivate enviroment
```bash
conda activate proenfo_env
conda deactivate
```

### Update your local environment

If there's a new package in the `proenfo_env.yml` file you have to update the packages in your local env

```bash
conda env update -f proenfo_env.yml
```

### Export your local environment

Export your environment for other users

```bash
conda env export > proenfo_env.yml 
```

### Recreate environment in connection with Pip
```bash
conda env remove --name proenfo_env
conda env create --file proenfo_env.yml
```

### Initial packages include
  - python=3.9.13
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - plotly
  - statsmodels
  - xlrd
  - jupyterlab
  - nodejs
  - mypy
  - pytorch
## Forecasting evaluation
We include several metrics to evaluate the forecasting performance, here is a visualization example. For details, you can check it in ./evaluation/metrics.py


![contents](https://github.com/Leo-VK/ProEnFo/tree/raw/main/figure/CT.png)

