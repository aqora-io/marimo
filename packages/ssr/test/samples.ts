import type { NotebookSnapshot } from "../src/types";

export const sampleNotebook: NotebookSnapshot = {
  notebook: {
    version: "1",
    cells: [
      {
        id: "Hbol",
        code: "from aqora_cli.pyarrow import dataset",
        code_hash: "8e97f755dc24461bd6dfed2f81b3a2d5",
        name: "_",
        config: {
          column: null,
          disabled: false,
          hide_code: false,
        },
      },
      {
        id: "MJUe",
        code: "import polars as pl",
        code_hash: "48f9ce6933b2bd65fba06745776b6bb9",
        name: "_",
        config: {
          column: null,
          disabled: false,
          hide_code: false,
        },
      },
      {
        id: "vblA",
        code: 'df = pl.scan_pyarrow_dataset(dataset("julian/titanic", "v0.1.0")).collect()\ndf',
        code_hash: "2a37a06aa817fff80c5b1287a3ce3ec5",
        name: "_",
        config: {
          column: null,
          disabled: false,
          hide_code: false,
        },
      },
      {
        id: "bkHC",
        code: "import marimo as mo",
        code_hash: "1d0db38904205bec4d6f6f6a1f6cec3e",
        name: "_",
        config: {
          column: null,
          disabled: false,
          hide_code: false,
        },
      },
      {
        id: "lEQa",
        code: 'mo.md(r"""\n# Hello, World!\n\nLorem ipsum dolor sit amet\n""")',
        code_hash: "895b10a20203c4cd29876db5a5a66586",
        name: "_",
        config: {
          column: null,
          disabled: false,
          hide_code: true,
        },
      },
      {
        id: "PKri",
        code: "mo.ui.button()",
        code_hash: "bfe8f22071f644c976b4c84c8b18f43c",
        name: "_",
        config: {
          column: null,
          disabled: false,
          hide_code: false,
        },
      },
    ],
    metadata: {
      marimo_version: "0.22.4",
    },
  },
  session: {
    version: "1",
    metadata: {
      marimo_version: "0.23.0",
      script_metadata_hash: "9080b0e87bbce128404c7713dde8ec95",
    },
    cells: [
      {
        id: "Hbol",
        code_hash: "8e97f755dc24461bd6dfed2f81b3a2d5",
        outputs: [
          {
            type: "data",
            data: {
              "text/plain": "",
            },
          },
        ],
        console: [],
      },
      {
        id: "MJUe",
        code_hash: "48f9ce6933b2bd65fba06745776b6bb9",
        outputs: [
          {
            type: "data",
            data: {
              "text/plain": "",
            },
          },
        ],
        console: [],
      },
      {
        id: "vblA",
        code_hash: "2a37a06aa817fff80c5b1287a3ce3ec5",
        outputs: [
          {
            type: "data",
            data: {
              "text/html":
                "<marimo-ui-element object-id='vblA-0' random-id='356563a5-7174-dc86-b055-fe4422c4547b'><marimo-table data-initial-value='[]' data-label='null' data-data='&quot;[{&#92;&quot;PassengerId&#92;&quot;:1,&#92;&quot;Survived&#92;&quot;:0,&#92;&quot;Pclass&#92;&quot;:3,&#92;&quot;Name&#92;&quot;:&#92;&quot;Braund, Mr. Owen Harris&#92;&quot;,&#92;&quot;Sex&#92;&quot;:&#92;&quot;male&#92;&quot;,&#92;&quot;Age&#92;&quot;:22.0,&#92;&quot;SibSp&#92;&quot;:1,&#92;&quot;Parch&#92;&quot;:0,&#92;&quot;Ticket&#92;&quot;:&#92;&quot;A/5 21171&#92;&quot;,&#92;&quot;Fare&#92;&quot;:7.25,&#92;&quot;Cabin&#92;&quot;:null,&#92;&quot;Embarked&#92;&quot;:&#92;&quot;S&#92;&quot;},{&#92;&quot;PassengerId&#92;&quot;:2,&#92;&quot;Survived&#92;&quot;:1,&#92;&quot;Pclass&#92;&quot;:1,&#92;&quot;Name&#92;&quot;:&#92;&quot;Cumings, Mrs. John Bradley (Florence Briggs Thayer)&#92;&quot;,&#92;&quot;Sex&#92;&quot;:&#92;&quot;female&#92;&quot;,&#92;&quot;Age&#92;&quot;:38.0,&#92;&quot;SibSp&#92;&quot;:1,&#92;&quot;Parch&#92;&quot;:0,&#92;&quot;Ticket&#92;&quot;:&#92;&quot;PC 17599&#92;&quot;,&#92;&quot;Fare&#92;&quot;:71.2833,&#92;&quot;Cabin&#92;&quot;:&#92;&quot;C85&#92;&quot;,&#92;&quot;Embarked&#92;&quot;:&#92;&quot;C&#92;&quot;},{&#92;&quot;PassengerId&#92;&quot;:3,&#92;&quot;Survived&#92;&quot;:1,&#92;&quot;Pclass&#92;&quot;:3,&#92;&quot;Name&#92;&quot;:&#92;&quot;Heikkinen, Miss. Laina&#92;&quot;,&#92;&quot;Sex&#92;&quot;:&#92;&quot;female&#92;&quot;,&#92;&quot;Age&#92;&quot;:26.0,&#92;&quot;SibSp&#92;&quot;:0,&#92;&quot;Parch&#92;&quot;:0,&#92;&quot;Ticket&#92;&quot;:&#92;&quot;STON/O2. 3101282&#92;&quot;,&#92;&quot;Fare&#92;&quot;:7.925,&#92;&quot;Cabin&#92;&quot;:null,&#92;&quot;Embarked&#92;&quot;:&#92;&quot;S&#92;&quot;},{&#92;&quot;PassengerId&#92;&quot;:4,&#92;&quot;Survived&#92;&quot;:1,&#92;&quot;Pclass&#92;&quot;:1,&#92;&quot;Name&#92;&quot;:&#92;&quot;Futrelle, Mrs. Jacques Heath (Lily May Peel)&#92;&quot;,&#92;&quot;Sex&#92;&quot;:&#92;&quot;female&#92;&quot;,&#92;&quot;Age&#92;&quot;:35.0,&#92;&quot;SibSp&#92;&quot;:1,&#92;&quot;Parch&#92;&quot;:0,&#92;&quot;Ticket&#92;&quot;:&#92;&quot;113803&#92;&quot;,&#92;&quot;Fare&#92;&quot;:53.1,&#92;&quot;Cabin&#92;&quot;:&#92;&quot;C123&#92;&quot;,&#92;&quot;Embarked&#92;&quot;:&#92;&quot;S&#92;&quot;},{&#92;&quot;PassengerId&#92;&quot;:5,&#92;&quot;Survived&#92;&quot;:0,&#92;&quot;Pclass&#92;&quot;:3,&#92;&quot;Name&#92;&quot;:&#92;&quot;Allen, Mr. William Henry&#92;&quot;,&#92;&quot;Sex&#92;&quot;:&#92;&quot;male&#92;&quot;,&#92;&quot;Age&#92;&quot;:35.0,&#92;&quot;SibSp&#92;&quot;:0,&#92;&quot;Parch&#92;&quot;:0,&#92;&quot;Ticket&#92;&quot;:&#92;&quot;373450&#92;&quot;,&#92;&quot;Fare&#92;&quot;:8.05,&#92;&quot;Cabin&#92;&quot;:null,&#92;&quot;Embarked&#92;&quot;:&#92;&quot;S&#92;&quot;},{&#92;&quot;PassengerId&#92;&quot;:6,&#92;&quot;Survived&#92;&quot;:0,&#92;&quot;Pclass&#92;&quot;:3,&#92;&quot;Name&#92;&quot;:&#92;&quot;Moran, Mr. James&#92;&quot;,&#92;&quot;Sex&#92;&quot;:&#92;&quot;male&#92;&quot;,&#92;&quot;Age&#92;&quot;:null,&#92;&quot;SibSp&#92;&quot;:0,&#92;&quot;Parch&#92;&quot;:0,&#92;&quot;Ticket&#92;&quot;:&#92;&quot;330877&#92;&quot;,&#92;&quot;Fare&#92;&quot;:8.4583,&#92;&quot;Cabin&#92;&quot;:null,&#92;&quot;Embarked&#92;&quot;:&#92;&quot;Q&#92;&quot;},{&#92;&quot;PassengerId&#92;&quot;:7,&#92;&quot;Survived&#92;&quot;:0,&#92;&quot;Pclass&#92;&quot;:1,&#92;&quot;Name&#92;&quot;:&#92;&quot;McCarthy, Mr. Timothy J&#92;&quot;,&#92;&quot;Sex&#92;&quot;:&#92;&quot;male&#92;&quot;,&#92;&quot;Age&#92;&quot;:54.0,&#92;&quot;SibSp&#92;&quot;:0,&#92;&quot;Parch&#92;&quot;:0,&#92;&quot;Ticket&#92;&quot;:&#92;&quot;17463&#92;&quot;,&#92;&quot;Fare&#92;&quot;:51.8625,&#92;&quot;Cabin&#92;&quot;:&#92;&quot;E46&#92;&quot;,&#92;&quot;Embarked&#92;&quot;:&#92;&quot;S&#92;&quot;},{&#92;&quot;PassengerId&#92;&quot;:8,&#92;&quot;Survived&#92;&quot;:0,&#92;&quot;Pclass&#92;&quot;:3,&#92;&quot;Name&#92;&quot;:&#92;&quot;Palsson, Master. Gosta Leonard&#92;&quot;,&#92;&quot;Sex&#92;&quot;:&#92;&quot;male&#92;&quot;,&#92;&quot;Age&#92;&quot;:2.0,&#92;&quot;SibSp&#92;&quot;:3,&#92;&quot;Parch&#92;&quot;:1,&#92;&quot;Ticket&#92;&quot;:&#92;&quot;349909&#92;&quot;,&#92;&quot;Fare&#92;&quot;:21.075,&#92;&quot;Cabin&#92;&quot;:null,&#92;&quot;Embarked&#92;&quot;:&#92;&quot;S&#92;&quot;},{&#92;&quot;PassengerId&#92;&quot;:9,&#92;&quot;Survived&#92;&quot;:1,&#92;&quot;Pclass&#92;&quot;:3,&#92;&quot;Name&#92;&quot;:&#92;&quot;Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)&#92;&quot;,&#92;&quot;Sex&#92;&quot;:&#92;&quot;female&#92;&quot;,&#92;&quot;Age&#92;&quot;:27.0,&#92;&quot;SibSp&#92;&quot;:0,&#92;&quot;Parch&#92;&quot;:2,&#92;&quot;Ticket&#92;&quot;:&#92;&quot;347742&#92;&quot;,&#92;&quot;Fare&#92;&quot;:11.1333,&#92;&quot;Cabin&#92;&quot;:null,&#92;&quot;Embarked&#92;&quot;:&#92;&quot;S&#92;&quot;},{&#92;&quot;PassengerId&#92;&quot;:10,&#92;&quot;Survived&#92;&quot;:1,&#92;&quot;Pclass&#92;&quot;:2,&#92;&quot;Name&#92;&quot;:&#92;&quot;Nasser, Mrs. Nicholas (Adele Achem)&#92;&quot;,&#92;&quot;Sex&#92;&quot;:&#92;&quot;female&#92;&quot;,&#92;&quot;Age&#92;&quot;:14.0,&#92;&quot;SibSp&#92;&quot;:1,&#92;&quot;Parch&#92;&quot;:0,&#92;&quot;Ticket&#92;&quot;:&#92;&quot;237736&#92;&quot;,&#92;&quot;Fare&#92;&quot;:30.0708,&#92;&quot;Cabin&#92;&quot;:null,&#92;&quot;Embarked&#92;&quot;:&#92;&quot;C&#92;&quot;}]&quot;' data-total-rows='891' data-total-columns='12' data-max-columns='50' data-banner-text='&quot;&quot;' data-pagination='true' data-page-size='10' data-field-types='[[&quot;PassengerId&quot;,[&quot;integer&quot;,&quot;i64&quot;]],[&quot;Survived&quot;,[&quot;integer&quot;,&quot;i64&quot;]],[&quot;Pclass&quot;,[&quot;integer&quot;,&quot;i64&quot;]],[&quot;Name&quot;,[&quot;string&quot;,&quot;str&quot;]],[&quot;Sex&quot;,[&quot;string&quot;,&quot;str&quot;]],[&quot;Age&quot;,[&quot;number&quot;,&quot;f64&quot;]],[&quot;SibSp&quot;,[&quot;integer&quot;,&quot;i64&quot;]],[&quot;Parch&quot;,[&quot;integer&quot;,&quot;i64&quot;]],[&quot;Ticket&quot;,[&quot;string&quot;,&quot;str&quot;]],[&quot;Fare&quot;,[&quot;number&quot;,&quot;f64&quot;]],[&quot;Cabin&quot;,[&quot;string&quot;,&quot;str&quot;]],[&quot;Embarked&quot;,[&quot;string&quot;,&quot;str&quot;]]]' data-show-filters='true' data-show-download='true' data-show-column-summaries='true' data-show-data-types='true' data-show-page-size-selector='true' data-show-column-explorer='true' data-show-chart-builder='true' data-row-headers='[]' data-has-stable-row-id='false' data-lazy='false' data-preload='false'></marimo-table></marimo-ui-element>",
            },
          },
        ],
        console: [],
      },
      {
        id: "bkHC",
        code_hash: "1d0db38904205bec4d6f6f6a1f6cec3e",
        outputs: [
          {
            type: "data",
            data: {
              "text/plain": "",
            },
          },
        ],
        console: [],
      },
      {
        id: "lEQa",
        code_hash: "895b10a20203c4cd29876db5a5a66586",
        outputs: [
          {
            type: "data",
            data: {
              "text/markdown":
                '<span class="markdown prose dark:prose-invert contents"><h1 id="hello-world">Hello, World!</h1>\n<span class="paragraph">Lorem ipsum dolor sit amet</span></span>',
            },
          },
        ],
        console: [],
      },
      {
        id: "PKri",
        code_hash: "bfe8f22071f644c976b4c84c8b18f43c",
        outputs: [
          {
            type: "data",
            data: {
              "text/html":
                "<marimo-ui-element object-id='JlJP-0' random-id='474efc73-6038-4472-02d9-8d8046975a10'><marimo-button data-initial-value='0' data-label='&quot;&lt;span class=&#92;&quot;markdown prose dark:prose-invert contents&#92;&quot;&gt;&lt;span class=&#92;&quot;paragraph&#92;&quot;&gt;click here&lt;/span&gt;&lt;/span&gt;&quot;' data-kind='&quot;neutral&quot;' data-disabled='false' data-full-width='false'></marimo-button></marimo-ui-element>",
            },
          },
        ],
        console: [],
      },
    ],
  },
};
