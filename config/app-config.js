/** @type {AppTypes.Config} */
window.config = {
  //orthancUrl: 'http://localhost:45821',
  routerBasename: '/viewer',
  showStudyList: true,
  extensions: [],
  modes: [],
  showWarningMessageForCrossOrigin: true,
  showCPUFallbackMessage: true,
  showLoadingIndicator: true,
  experimentalStudyBrowserSort: false,
  strictZSpacingForVolumeViewport: true,
  studyPrefetcher: {
    enabled: true,
    displaySetsCount: 2,
    maxNumPrefetchRequests: 10,
    order: 'closest',
  },
  // The custom DisclaimerBanner (view-ai-result) replaces OHIF's own
  // investigational-use dialog; without this the two stack on top of each other.
  investigationalUseDialog: {
    option: 'never',
  },
  defaultDataSourceName: 'dicomweb',
  studyList: {
    defaultSortField: 'StudyDate',
    defaultSortOrder: 'descending',
    defaultTimeRange: 'last7days',
    timeRanges: [
      { label: 'Last 7 days', value: 'last7days' },
      { label: 'Last 30 days', value: 'last30days' },
      { label: 'Last 90 days', value: 'last90days' },
      { label: 'Last year', value: 'lastyear' },
      { label: 'All time', value: 'all' },
    ],
  },
  dataSources: [
    {
      namespace: '@ohif/extension-default.dataSourcesModule.dicomweb',
      sourceName: 'dicomweb',
      configuration: {
        friendlyName: 'Orthanc Server',
        name: 'Orthanc',
        wadoUriRoot: '/wado',
        qidoRoot: '/pacs/dicom-web',
        wadoRoot: '/pacs/dicom-web',
        qidoSupportsIncludeField: true,
        supportsReject: true,
        imageRendering: 'wadors',
        thumbnailRendering: 'wadors',
        enableStudyLazyLoad: true,
        supportsFuzzyMatching: true,
        supportsWildcard: true,
        dicomUploadEnabled: true,
        omitQuotationForMultipartRequest: true,
      },
    }
  ],
  // Default trained model/router pairs; names include the weight release.
  // Keep mst-ai and its URL stable so existing saved selections still work.
  aiEndpoints: [
    {
      id: 'mst-ai',
      name: 'MST swarm May 2026',
      url: 'http://orthanc-router-mst:8042/dicom-web',
    },
    {
      id: 'agaldran-ai',
      name: 'agaldran swarm May 2026',
      url: 'http://orthanc-router-odelia-agaldran:8042/dicom-web',
    },
    {
      id: 'swinunetr-ai',
      name: 'SwinUNETR swarm May 2026',
      url: 'http://orthanc-router-odelia-bcn-aim:8042/dicom-web',
    },
    {
      id: 'abmil-ai',
      name: 'ABMIL swarm May 2026',
      url: 'http://orthanc-router-odelia-lme-abmil:8042/dicom-web',
    },
    {
      id: 'meisenmeister-ai',
      name: 'MeisenMeister swarm August 2026',
      url: 'http://orthanc-router-odelia-divide-and-conquer:8042/dicom-web',
    },
  ],
  httpErrorHandler: error => {
    console.warn(`HTTP Error Handler (status: ${error.status})`, error);
  },
  oidc: [
    {
      authority: '/keycloak/realms/ohif',
      client_id: 'ohif_viewer',
      redirect_uri: '/viewer/callback',
      scope: 'openid profile email',
      post_logout_redirect_uri: '/viewer/',
      response_type: 'code',
      // Disable features that cause Firefox rate limiting
      automaticSilentRenew: false,
      monitorSession: false,
      revokeAccessTokenOnSignout: true
    }],
};
