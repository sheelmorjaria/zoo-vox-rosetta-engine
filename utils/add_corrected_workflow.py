"""
Helper script to add corrected workflow to species analyzers.
"""

# Template for adding corrected workflow to each species
CORRECTED_WORKFLOW_TEMPLATE = '''
    def run_corrected_workflow(
        self,
        max_files: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> RosettaStoneResults:
        """
        Run corrected Rosetta Stone analysis with harmonic affirmation
        and compositional validation.
        """
        from harmonic_affirmation import HarmonicAffirmationAnalyzer
        from compositional_validation import CompositionalValidator
        import json

        logger.info("=" * 60)
        logger.info(f"{self.__class__.__name__.replace('RosettaStone', '').upper()} ROSETTA STONE ANALYSIS (CORRECTED WORKFLOW)")
        logger.info("=" * 60)

        output_path = Path(output_dir or self.config.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        results = RosettaStoneResults(
            species=self.config.species,
            timestamp=datetime.now().isoformat(),
            config=self.config.__dict__,
            atomic_words=None,
            microharmonic=None,
            sentences=None
        )

        # Phase 1: Atomic Word Discovery
        logger.info("\\n" + "=" * 60)
        logger.info("PHASE 1: ATOMIC WORD DISCOVERY")
        logger.info("=" * 60)

        phrase_library = self.atomic_word_discovery.build_phrase_library(max_files=max_files)

        results.atomic_words = AtomicWordResults(
            vocabulary_size=len(phrase_library),
            total_occurrences=sum([p['total_occurrences'] for p in phrase_library.values()]),
            phrase_types=list(phrase_library.keys())[:50],
            phrase_library=phrase_library
        )

        logger.info(f"Atomic Word Types: {results.atomic_words.vocabulary_size}")

        # Phase 1.5: Harmonic Affirmation
        if self.config.use_harmonic_affirmation:
            logger.info("\\n" + "=" * 60)
            logger.info("PHASE 1.5: HARMONIC AFFIRMATION")
            logger.info("=" * 60)

            harmonic_analyzer = HarmonicAffirmationAnalyzer(config=self.config)
            audio_files = list(self.audio_dir.rglob("*.wav"))[:max_files] if max_files else list(self.audio_dir.rglob("*.wav"))

            affirmed_clusters = harmonic_analyzer.affirm_atomic_words(audio_files)
            logger.info(f"Affirmed {len(affirmed_clusters)} harmonic clusters")

        # Phase 2: Sentence Discovery
        logger.info("\\n" + "=" * 60)
        logger.info("PHASE 2: SENTENCE DISCOVERY")
        logger.info("=" * 60)

        # ... sentence discovery code ...

        # Phase 2.5: Compositional Validation
        if self.config.use_compositional_validation:
            logger.info("\\n" + "=" * 60)
            logger.info("PHASE 2.5: COMPOSITIONAL VALIDATION")
            logger.info("=" * 60)

            validator = CompositionalValidator(config=self.config)
            validation_report = validator.validate_sentence_dataset(
                list(self.audio_dir.rglob("*.wav"))[:max_files] if max_files else list(self.audio_dir.rglob("*.wav")),
                phrase_library,
                max_files=max_files
            )

            logger.info(f"Validation Rate: {validation_report.validation_rate*100:.1f}%")

        # Phase 3: Superposition Detection
        logger.info("\\n" + "=" * 60)
        logger.info("PHASE 3: SUPERPOSITION DETECTION")
        logger.info("=" * 60)

        # ... superposition detection code ...

        return results
'''

print("Template generated. Each species needs custom implementation.")
